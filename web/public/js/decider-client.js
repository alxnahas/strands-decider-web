// Main-thread handle to the inference worker. Monotonic request ids; one warm session.
export class Decider {
  constructor({ device = "webgpu", model = "q4f16p", assets = null, onEvent = () => {} } = {}) {
    const url = new URL("./worker.js", import.meta.url); url.searchParams.set("model", model); for (const [k, v] of new URLSearchParams(location.search)) if (k !== "model") url.searchParams.set(k, v);
    if (assets && !url.searchParams.has("assets")) url.searchParams.set("assets", assets);  // model files on another origin
    this.worker = new Worker(url, { type: "module" });
    this.pending = new Map(); this.nextId = 1; this.onEvent = onEvent;
    this.ready = new Promise((resolve, reject) => { this._ready = { resolve, reject }; });
    this.worker.onmessage = ({ data }) => {
      if (data.type === "ready") { this.info = data; this._ready.resolve(data); }
      else if (data.type === "result") { this.pending.get(data.id)?.resolve(data); this.pending.delete(data.id); }
      else if (data.type === "error") {
        if (data.id && this.pending.has(data.id)) { this.pending.get(data.id).reject(new Error(data.error)); this.pending.delete(data.id); }
        else this._ready.reject(new Error(data.error));
      }
      this.onEvent(data);
    };
    this.worker.onerror = (e) => this._ready.reject(new Error(e.message));
    this.worker.postMessage({ type: "init", device });
  }
  async decideMany(state, questions) {
    await this.ready;
    const id = this.nextId++;
    return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.worker.postMessage({ type: "decideMany", id, state, questions }); });
  }
  async bench(lengths) {
    await this.ready; const id = this.nextId++;
    return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.worker.postMessage({ type: "bench", id, lengths }); });
  }
  /** Fetch the embedding rows [{state, question}, ...] will need (a lazy-embedding build), without deciding. */
  async prefetch(items) {
    await this.ready; const id = this.nextId++;
    return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.worker.postMessage({ type: "prefetch", id, items }); });
  }
  async decide(state, question) {
    await this.ready;
    const id = this.nextId++;
    return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.worker.postMessage({ type: "decide", id, state, question }); });
  }
}
