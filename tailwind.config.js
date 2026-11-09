const tok = (v) => `color-mix(in srgb, var(${v}) calc(<alpha-value> * 100%), transparent)`;

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: tok("--bg"),
        surface: tok("--surface"),
        "surface-2": tok("--surface-2"),
        line: tok("--line"),
        ink: tok("--ink"),
        muted: tok("--muted"),
        local: tok("--local"),
        global: tok("--global"),
        vector: tok("--vector"),
        good: tok("--good"),
        warn: tok("--warn"),
        bad: tok("--bad"),
      },
      fontFamily: {
        title: ["Archivo", "system-ui", "sans-serif"],
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      borderRadius: { card: "10px" },
    },
  },
  plugins: [],
};
