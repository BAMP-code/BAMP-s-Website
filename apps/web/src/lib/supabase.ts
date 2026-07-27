import { PostgrestClient } from "@supabase/postgrest-js";

// Fetched once, at build time (astro build runs this module's frontmatter
// imports server-side). The site is static output — there's no runtime
// server, so nothing here ever touches a browser or exposes a secret key;
// SUPABASE_ANON_KEY is Supabase's public, RLS-scoped key by design.
//
// PostgrestClient (not the full supabase-js) because we only ever do
// read-only table queries here — no auth, storage, or realtime, so no
// reason to pull in a WebSocket-dependent client for a static build.

const url = import.meta.env.SUPABASE_URL;
const anonKey = import.meta.env.SUPABASE_ANON_KEY;

if (!url || !anonKey) {
  throw new Error(
    "Missing SUPABASE_URL or SUPABASE_ANON_KEY. Set them in apps/web/.env " +
      "(see .env.example) and in the Vercel project's environment variables.",
  );
}

export const supabase = new PostgrestClient(`${url}/rest/v1`, {
  headers: { apikey: anonKey, Authorization: `Bearer ${anonKey}` },
  schema: "public",
});
