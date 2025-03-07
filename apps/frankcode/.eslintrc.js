module.exports = {
    env: {
      node: true,
      es6: true,
      jest: true
    },
    extends: [
      'eslint:recommended',
      'plugin:node/recommended',
      'plugin:prettier/recommended'
    ],
    parserOptions: {
      ecmaVersion: 2020
    },
    rules: {
      'no-console': 'off',
      'node/no-unsupported-features/es-syntax': [
        'error',
        { ignores: ['modules'] }
      ],
      'prettier/prettier': 'warn',
      'node/no-unpublished-require': 'off'
    }
  };