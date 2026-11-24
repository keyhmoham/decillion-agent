import { wsUrl } from "../api/client";
import type { ServerEvent } from "../api/types";

/** Reconnecting WebSocket bound to one session. Resumes from the last seen event sequence. */
export class SessionSocket {
  private ws?: WebSocket;
  private closed = false;
  private retry = 0;
  private lastSeq: number;
  constructor(
    private sid: string,
    since: number,
    private onEvent: (e: ServerEvent) => void,
    private onState: (connected: boolean) => void,
  ) {
    this.lastSeq = since;
    this.connect();
  }

  private connect() {
    if (this.closed) return;
    const ws = new WebSocket(wsUrl(this.sid, this.lastSeq));
    this.ws = ws;
    ws.onopen = () => { this.retry = 0; this.onState(true); };
    ws.onmessage = (m) => {
      try {
        const e = JSON.parse(m.data) as ServerEvent;
        if (e.seq) this.lastSeq = Math.max(this.lastSeq, e.seq);
        this.onEvent(e);
      } catch { /* ignore malformed frame */ }
    };
    ws.onclose = () => {
      this.onState(false);
      if (this.closed) return;
      const delay = Math.min(10000, 500 * 2 ** this.retry++);
      setTimeout(() => this.connect(), delay);
    };
    ws.onerror = () => ws.close();
  }

  send(msg: Record<string, unknown>): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
      return true;
    }
    return false;
  }

  close() {
    this.closed = true;
    this.ws?.close();
  }
}
