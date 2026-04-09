import terser from "@rollup/plugin-terser";
import { nodeResolve } from "@rollup/plugin-node-resolve";

const plugins = (browser) => [nodeResolve({ browser }), terser({ format: { comments: false } })];

const OUTPUT_CONFIGS = [
  // Node versions
  {
    file: "./dist/kokoro.cjs",
    format: "cjs",
  },
  {
    file: "./dist/kokoro.js",
    format: "esm",
  },

  // Web version (uses dir output because ephone lang packs are dynamic imports → multiple chunks)
  {
    dir: "./dist",
    entryFileNames: "kokoro.web.js",
    chunkFileNames: "kokoro.web-[hash].js",
    format: "esm",
  },
];

const WEB_SPECIFIC_CONFIG = {
  onwarn: (warning, warn) => {
    if (!warning.message.includes("@huggingface/transformers")) warn(warning);
  },
};

const NODE_SPECIFIC_CONFIG = {
  external: ["@huggingface/transformers", "ephone"],
};

export default OUTPUT_CONFIGS.map((output) => {
  const web = (output.file ?? output.entryFileNames ?? "").includes(".web.");
  return {
    input: "./src/kokoro.js",
    output,
    plugins: plugins(web),
    ...(web ? WEB_SPECIFIC_CONFIG : NODE_SPECIFIC_CONFIG),
  };
});
