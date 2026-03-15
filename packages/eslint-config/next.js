/** @type {import("eslint").Linter.Config} */
module.exports = {
  extends: [
    "next/core-web-vitals",
    "plugin:jsx-a11y/recommended",
    "plugin:import/recommended",
    "plugin:import/typescript",
  ],
  rules: {
    "jsx-a11y/alt-text": "error",
    "jsx-a11y/aria-props": "error",
    "jsx-a11y/aria-role": "error",
    "jsx-a11y/role-has-required-aria-props": "error",
    // Carousels are interactive regions that need keyboard handling
    "jsx-a11y/no-noninteractive-element-interactions": ["error", {
      handlers: ["onClick"],
      body: ["onKeyDown"],
    }],
    "jsx-a11y/no-noninteractive-tabindex": ["warn", {
      tags: ["section"],
      roles: ["tabpanel"],
      allowExpressionValues: true,
    }],
  },
};
