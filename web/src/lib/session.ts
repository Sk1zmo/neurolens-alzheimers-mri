"use client";

/**
 * Anonymous, device-local identity.
 *
 * There are no accounts. A random id in localStorage is what ties a set of
 * uploads together into "your history". It is deliberately not a fingerprint:
 * clearing site data starts a fresh identity, and the id carries no personal
 * information of any kind.
 */

const KEY = "neurolens.session";

export function getSessionId(): string {
  if (typeof window === "undefined") return "";
  let id = window.localStorage.getItem(KEY);
  if (!id) {
    id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `s_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
    window.localStorage.setItem(KEY, id);
  }
  return id;
}

export function resetSessionId(): string {
  if (typeof window === "undefined") return "";
  window.localStorage.removeItem(KEY);
  return getSessionId();
}
