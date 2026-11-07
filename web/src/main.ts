import './styles.css';
/**
 * Peter AI — Web client entry point.
 *
 * Routing is intentionally minimal: a hash router so the app works
 * without a server re-writing URLs.
 *
 * Routes:
 *   #/login   → login page
 *   #/app     → main app (chat + mind view)  [requires auth]
 *   (default) → redirect to #/login or #/app based on stored token
 */
import { getStoredToken } from './auth/token';

function route() {
  const hash = window.location.hash || '';
  const token = getStoredToken();

  if (hash.startsWith('#/app') && !token) {
    window.location.hash = '#/login';
    return;
  }
  if ((hash === '' || hash === '#/' || hash === '#/login') && token) {
    window.location.hash = '#/app';
    return;
  }

  if (hash.startsWith('#/app')) {
    import('./pages/app').then(m => m.mount(document.getElementById('app')!));
  } else {
    import('./pages/login').then(m => m.mount(document.getElementById('app')!));
  }
}

window.addEventListener('hashchange', () => route());
route();
