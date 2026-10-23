import { apiFetch } from '../api/client';
import { storeToken } from '../auth/token';

export function mount(root: HTMLElement): void {
  root.innerHTML = `
    <div class="login-page">
      <div class="login-bg-glow"></div>
      <div class="login-card">
        <div class="login-logo">
          <div class="login-logo-icon">🧠</div>
          <div>
            <div class="login-logo-text">Peter AI</div>
            <div class="login-logo-sub">Self-learning intelligence</div>
          </div>
        </div>
        <form id="login-form">
          <div class="login-field">
            <label class="login-label" for="email">Email</label>
            <input id="email" class="login-input" type="email" autocomplete="username"
                   placeholder="you@example.com" required />
          </div>
          <div class="login-field">
            <label class="login-label" for="password">Password</label>
            <input id="password" class="login-input" type="password"
                   autocomplete="current-password" placeholder="••••••••" required />
          </div>
          <button type="submit" class="login-btn">Sign in →</button>
          <div id="login-error" class="login-error hidden"></div>
        </form>
        <div class="login-footer">Peter learns from every conversation · data stays local</div>
      </div>
    </div>
  `;

  const form  = root.querySelector<HTMLFormElement>('#login-form')!;
  const errEl = root.querySelector<HTMLElement>('#login-error')!;
  const btn   = root.querySelector<HTMLButtonElement>('.login-btn')!;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errEl.classList.add('hidden');
    btn.textContent = 'Signing in…';
    btn.disabled    = true;

    const email    = (root.querySelector<HTMLInputElement>('#email')!).value.trim();
    const password = (root.querySelector<HTMLInputElement>('#password')!).value;

    try {
      const res = await apiFetch('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        errEl.textContent = body.detail ?? 'Login failed. Check your credentials.';
        errEl.classList.remove('hidden');
        return;
      }
      const { access_token } = await res.json();
      storeToken(access_token);
      window.location.hash = '#/app';
    } catch {
      errEl.textContent = 'Network error — is the server running?';
      errEl.classList.remove('hidden');
    } finally {
      btn.textContent = 'Sign in →';
      btn.disabled    = false;
    }
  });
}
