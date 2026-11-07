/**
 * WebSocket client for the live mind event stream (/ws/mind).
 *
 * - One connection per app session (shared across all panels)
 * - Reconnects with exponential backoff when the socket closes
 * - `listeners` is public so the replay panel can drive the same handlers
 */
import { getStoredToken } from '../auth/token';

export type MindEvent = {
  type:    string;
  user_id: string;
  turn_id: string;
  ts?:     number;
  [key: string]: unknown;
};

type Listener = (ev: MindEvent) => void;

export class PeterSocket {
  /** Public so the replay panel can drive the same handlers */
  readonly listeners: Listener[] = [];

  private ws:       WebSocket | null = null;
  private _backoff: number           = 1000;

  constructor() { this._connect(); }

  /** Register a listener; returns an unsubscribe function */
  onEvent(fn: Listener): () => void {
    this.listeners.push(fn);
    return () => {
      const i = this.listeners.indexOf(fn);
      if (i !== -1) this.listeners.splice(i, 1);
    };
  }

  private _connect(): void {
    const token = getStoredToken();
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url   = `${proto}://${location.host}/ws/mind?token=${encodeURIComponent(token ?? '')}`;

    this.ws = new WebSocket(url);

    this.ws.onmessage = (e: MessageEvent) => {
      try {
        const ev: MindEvent = JSON.parse(e.data as string);
        if (ev.type === 'ping') return;          // keepalive — ignore
        this.listeners.forEach(fn => { try { fn(ev); } catch { /* ignore */ } });
      } catch { /* malformed frame */ }
    };

    this.ws.onclose = () => {
      if (!getStoredToken()) return;             // signed out — stop reconnecting
      setTimeout(() => {
        this._backoff = Math.min(this._backoff * 2, 30_000);
        this._connect();
      }, this._backoff);
    };

    this.ws.onopen = () => { this._backoff = 1000; };
  }

  close(): void { this.ws?.close(); }
}
