import { Client } from "@langchain/langgraph-sdk";

const client = new Client({ apiUrl: "http://127.0.0.1:3265" });
const thread = await client.threads.create();
console.log("thread", thread.thread_id);

const stream = client.runs.stream(thread.thread_id, "agent", {
  input: {
    messages: [
      { type: "human", content: "Please research the latest AI agent progress and summarize key developments in 2025." }
    ],
    mode: "research"
  },
  streamMode: ["updates", "custom", "values"]
});

let count = 0;
try {
  for await (const chunk of stream) {
    count += 1;
    console.log("chunk", count, JSON.stringify(chunk).slice(0, 400));
    if (count >= 12) {
      break;
    }
  }
  console.log("stream-checkpoint-ok", count);
} catch (err) {
  console.error("stream-error", err?.stack || err);
  process.exitCode = 1;
}
