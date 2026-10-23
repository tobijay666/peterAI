/**
 * Fetch wrapper that attaches the Bearer token to every request.
 * All API calls go through here so the token is never duplicated.
 */
import { getStoredToken, clearToken } from '../auth/token';

const BASE = '';  // same origin; vite dev proxy forwards /api → :8000

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const token = getStoredToken();
  const headers = new Headers(init.headers ?? {});
  headers.set('Content-Type', 'application/json');
  if (token) headers.set('Authorization', `Bearer ${token}`);

  const res = await fetch(`${BASE}${path}`, { ...init, headers });

  if (res.status === 401) {
    // Token expired or invalid — send user back to login
    clearToken();
    window.location.hash = '#/login';
  }
  return res;
}
