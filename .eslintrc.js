module.exports = {
  env: {
    browser: true,
    es2021: true,
  },
  globals: {
    defineAppConfig: "readonly",
    definePageConfig: "readonly",
    API: "readonly",
    ToolTypes: "readonly",
    NodeJS: "readonly",
  },
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: "latest",
    sourceType: "module",
  },
};
