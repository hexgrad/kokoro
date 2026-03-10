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

  // Web version
  {
    file: "./dist/kokoro.web.js",
    format: "esm",
  },
];

const WEB_SPECIFIC_CONFIG = {
  onwarn: (warning, warn) => {
    if (!warning.message.includes("@huggingface/transformers")) warn(warning);
  },
};

const NODE_SPECIFIC_CONFIG = {
  external: ["@huggingface/transformers"],
};

export default OUTPUT_CONFIGS.map((output) => {
  const web = output.file.endsWith(".web.js");
  return {
    input: "./src/kokoro.js",
    output,
    plugins: plugins(web),
    ...(web ? WEB_SPECIFIC_CONFIG : NODE_SPECIFIC_CONFIG),
  };
});
