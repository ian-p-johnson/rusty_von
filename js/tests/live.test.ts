import { describe, expect, it } from "bun:test";
import { VonClient, choice, noul, score, VonError } from "../src/index.js";

// Live wire-contract tests against a running von server (either runtime).
// Skipped entirely unless VON_LIVE_URL is set, so `bun test` stays green
// standalone:
//
//   VON_LIVE_URL=http://127.0.0.1:8002 VON_LIVE_KEY=golden-test-key bun test
//
const LIVE_URL = process.env.VON_LIVE_URL;
const LIVE_KEY = process.env.VON_LIVE_KEY;
const d = LIVE_URL ? describe : describe.skip;

d("Von SDK against a live server", () => {
  const client = new VonClient({ baseURL: LIVE_URL, apiKey: LIVE_KEY });

  it("reports health", async () => {
    const res = await fetch(`${LIVE_URL}/health`);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("ok");
    expect(body.service).toBe("von-decision-server");
  });

  it("lists models", async () => {
    const res = await fetch(`${LIVE_URL}/v1/models`);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.object).toBe("list");
    expect(body.models.map((m: any) => m.name)).toContain("von-latest");
  });

  it("evaluates a Choice question end-to-end", async () => {
    const resp = await client.systemOne({
      state: "Customer requests refund for duplicate charge",
      questions: {
        decision: choice("Which one?", {
          billing: "Invoices and refunds",
          technical: "Bugs and outages",
        }),
      },
    });
    expect(resp.model).toBe("von-1.1.0");
    const ans = resp.answers.decision as any;
    expect(ans.type).toBe("choice");
    expect(["billing", "technical"]).toContain(ans.choice);
    expect(ans.confidence).toBeGreaterThanOrEqual(0);
    expect(ans.confidence).toBeLessThanOrEqual(1);
    expect(resp.usage.output_tokens).toBe(1);
  });

  it("stamps the served model even for old model ids", async () => {
    const resp = await client.systemOne({
      state: "hello",
      questions: { q: noul("Is this fine?") },
      model: "von-1.0" as any,
    });
    expect(resp.model).toBe("von-1.1.0");
  });

  it("evaluates a Noul question", async () => {
    const prob = await client.judge(
      "The server is returning 500s for every request.",
      "Is the server down?",
      "Requests fail with errors",
      "Requests succeed normally",
    );
    expect(prob).toBeGreaterThanOrEqual(0);
    expect(prob).toBeLessThanOrEqual(1);
  });

  it("evaluates a Score question", async () => {
    const ans = await client.rate("Mild latency increase", ["Low", "Medium", "High"]);
    expect(ans.type).toBe("score");
    expect(Object.keys(ans.legend).sort()).toEqual(["0", "1", "2"]);
    expect(ans.score).toBeGreaterThanOrEqual(0);
  });

  it("handles structured (nested dict) state", async () => {
    const resp = await client.systemOne({
      state: { host: "db-1", port: 5432, replicas: [1, 2], healthy: null },
      questions: { q: choice("Pick", { a: null, b: null }) },
    });
    const ans = resp.answers.q as any;
    expect(["a", "b"]).toContain(ans.choice);
  });

  it("enforces the bearer-key contract", async () => {
    const keyless = new VonClient({ baseURL: LIVE_URL });
    await expect(
      keyless.systemOne({ state: "x", questions: { q: noul("ok?") } })
    ).rejects.toMatchObject({ status: 401 });

    const wrong = new VonClient({ baseURL: LIVE_URL, apiKey: "wrong-key" });
    await expect(
      wrong.systemOne({ state: "x", questions: { q: noul("ok?") } })
    ).rejects.toMatchObject({ status: 401 });
  });

  it("rejects malformed JSON with 422", async () => {
    const res = await fetch(`${LIVE_URL}/v1/systemone`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(LIVE_KEY ? { Authorization: `Bearer ${LIVE_KEY}` } : {}),
      },
      body: '{"model": "von-latest"',
    });
    expect(res.status).toBe(422);
    const body = await res.json();
    expect(body.detail[0].type).toBe("json_invalid");
  });

  it("raises VonError on unknown question types", async () => {
    await expect(
      client.systemOne({
        state: "x",
        questions: { q: { type: "wat", instructions: "x", criteria: null } as any },
      })
    ).rejects.toBeInstanceOf(VonError);
  });

  it("decide helper round-trips", async () => {
    const ans = await client.decide("Charge me twice!", ["refund", "tech"]);
    expect(["refund", "tech"]).toContain(ans.choice);
  });
});
