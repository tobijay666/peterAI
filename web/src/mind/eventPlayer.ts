/**
 * EventPlayer — plays mind events one after another with minimum durations,
 * so a burst of 40 sleep events reads as a sequence instead of a flicker.
 * The live socket and the replay panel feed the same player.
 *
 * The handler returns how long (ms) the event's animation needs. When a
 * backlog builds up the player speeds up instead of falling behind forever.
 */
import type { MindEvent } from '../live/socket';

export type Handler = (ev: MindEvent) => number | void;

export class EventPlayer {
  private queue: MindEvent[] = [];
  private busy = false;
  private stopped = false;

  constructor(private handle: Handler) {}

  push(ev: MindEvent): void {
    if (this.stopped) return;
    this.queue.push(ev);
    if (!this.busy) void this.drain();
  }

  get backlog(): number { return this.queue.length; }

  stop(): void { this.stopped = true; this.queue = []; }

  private speed(): number {
    const n = this.queue.length;
    return n > 120 ? 0.2 : n > 50 ? 0.5 : n > 15 ? 0.8 : 1;
  }

  private async drain(): Promise<void> {
    this.busy = true;
    while (this.queue.length && !this.stopped) {
      const ev = this.queue.shift()!;
      let ms = 0;
      try { ms = Number(this.handle(ev) ?? 0); } catch (e) { console.warn('[Mind] event handler failed', ev.type, e); }
      const wait = ms * this.speed();
      if (wait > 0) await new Promise(r => setTimeout(r, wait));
    }
    this.busy = false;
  }
}
