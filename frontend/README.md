# Frontend for the system



## How to use:

```
npm install .(first time using) 
npm run build 
npm run start
```



## Note:

If you are running the run build command for the first time and you have not installed ESLint, a compilation error will occur，for example:

```powershell

PS D:\workshop\model_attack_system\frontend\starter-kit> npm run build
> vuexy-mui-nextjs-admin-template@3.1.0 build
> next build

Attention: Next.js now collects completely anonymous telemetry regarding usage.
This information is used to shape Next.js' roadmap and prioritize features.
You can learn more, including how to opt-out if you'd not like to participate in this anonymous program, by visiting the following URL:
https://nextjs.org/telemetry

  ²% Next.js 14.2.5

   Creating an optimized production build ...

 ' Compiled successfully

Failed to compile.

./src/app/(dashboard)/home/FileUploader.tsx
3:1  Error: There should be at least one empty line between import groups  import/order

./src/app/(dashboard)/home/page.tsx
3:1  Error: There should be at least one empty line between import groups  import/order
32:5  Error: Expected blank line before this statement.  padding-line-between-statements

./src/components/layout/shared/Logo.tsx
14:8  Error: 'VuexyLogo' is defined but never used.  @typescript-eslint/no-unused-vars

info  - Need to disable some ESLint rules? Learn more here: https://nextjs.org/docs/basic-features/eslint#disabling-rules
   Linting and checking validity of types  .

```

If you don't have a log check required or Fix code style manually, ignore it .eslintrc.js the build and modify its file content to:

```js
module.exports{
...
  "rules": {
    ...
    "@typescript-eslint/no-unused-vars": "off",
    ...
    'import/order': [
      // 'error',
      'off',
    ...
  }

}
```

Relaxation of import/order rules:

```js
'import/order': [
  'error',
  {
    groups: ['builtin', 'external', 'internal', 'parent', 'sibling', 'index'],
    pathGroupsExcludedImportTypes: ['react'],
    'newlines-between': 'always', // 可以改为 'off' 来禁用空行检查
  },
],
```

and adjust` padding-line-between-statements：`

```js
'padding-line-between-statements': [
  'error',
  { blankLine: 'always', prev: ['const', 'let', 'var'], next: '*' },
  { blankLine: 'always', prev: '*', next: ['return'] },
],
```

"vuexy-mui-nextjs-admin-template": "link" when using "npm install" error

