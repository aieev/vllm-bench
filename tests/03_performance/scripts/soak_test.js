import sse from "k6/x/sse";
import { check } from "k6";
import { Trend, Counter, Rate } from "k6/metrics";

const llm_ttft = new Trend("llm_ttft", true);
const llm_tps = new Trend("llm_tps");
const llm_tbt = new Trend("llm_tbt", true);
const llm_total_tokens = new Counter("llm_total_tokens");
const llm_errors = new Counter("llm_errors");
const llm_error_rate = new Rate("llm_error_rate");

const BASE_URL = __ENV.VLLM_BASE_URL || "http://localhost:8000/v1";
const API_KEY = __ENV.VLLM_API_KEY || "";
const MODEL = __ENV.MODEL_NAME || "default";
const SOAK_DURATION = __ENV.SOAK_DURATION || "30m";

export const options = {
  scenarios: {
    soak_test: {
      executor: "constant-vus",
      vus: 5,
      duration: SOAK_DURATION,
    },
  },
  thresholds: {
    llm_error_rate: ["rate<0.01"],
    llm_ttft: ["p(99)<10000"],
  },
};

const PROMPTS = [
  "Tell me a short story about a robot.",
  "Explain how a CPU works in simple terms.",
  "Write a haiku about the ocean.",
  "Describe the process of photosynthesis.",
  "What are three interesting facts about space?",
  "Summarize the theory of relativity.",
  "How does a compiler work?",
  "What is the difference between TCP and UDP?",
];

export default function () {
  const url = `${BASE_URL}/chat/completions`;
  const prompt = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];

  const payload = JSON.stringify({
    model: MODEL,
    messages: [{ role: "user", content: prompt }],
    stream: true,
    max_tokens: 256,
    temperature: 0.0,
    chat_template_kwargs: { enable_thinking: false },
  });

  const params = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: payload,
  };

  let tokenCount = 0;
  let firstTokenTime = null;
  let lastTokenTime = null;
  const requestStart = Date.now();
  let prevTokenTime = null;
  let errorOccurred = false;

  const response = sse.open(url, params, function (client) {
    client.on("event", function (event) {
      if (!event.data || event.data.trim() === "[DONE]") {
        client.close();
        return;
      }

      try {
        const parsed = JSON.parse(event.data);
        const delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta;
        if (delta && delta.content) {
          const now = Date.now();

          if (firstTokenTime === null) {
            firstTokenTime = now;
            llm_ttft.add(now - requestStart);
          }

          tokenCount++;
          lastTokenTime = now;

          if (prevTokenTime !== null) {
            llm_tbt.add(now - prevTokenTime);
          }
          prevTokenTime = now;
        }
      } catch (e) {
        // skip non-JSON lines
      }
    });

    client.on("error", function (e) {
      errorOccurred = true;
      llm_errors.add(1);
      llm_error_rate.add(1);
      client.close();
    });
  });

  if (errorOccurred || response.status !== 200) {
    if (!errorOccurred) {
      llm_errors.add(1);
      llm_error_rate.add(1);
    }
    return;
  }

  llm_error_rate.add(0);

  check(response, {
    "status is 200": (r) => r.status === 200,
  });

  if (tokenCount > 0 && firstTokenTime !== null && lastTokenTime !== null) {
    const generationTimeMs = lastTokenTime - firstTokenTime;
    if (generationTimeMs > 0) {
      const tps = (tokenCount / generationTimeMs) * 1000;
      llm_tps.add(tps);
    }
    llm_total_tokens.add(tokenCount);
  }
}
