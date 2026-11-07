/**
 * Thin wrapper around sessionStorage for the JWT/Supabase access token.
 * Using sessionStorage (not localStorage) so the token is cleared when
 * the tab closes — safer for a shared machine.
 */
const KEY = 'peter_access_token';

export function storeToken(token: string): void {
  try { sessionStorage.setItem(KEY, token); } catch { /* private window */ }
}

export function getStoredToken(): string | null {
  try { return sessionStorage.getItem(KEY); } catch { return null; }
}

export function clearToken(): void {
  try { sessionStorage.removeItem(KEY); } catch { /* ignore */ }
}
