# Publishing the static showcase

Only the allowlisted output of `scripts/build-site.js` is publishable. Never host
`src/server.js`, the lab API, local profiles, feedback databases, model weights,
or the entire checkout on a public server. The hosted `/lab/` is a setup guide,
not a working public inference or tracking service.

## GitHub Pages

The manually dispatched **Publish static showcase** workflow validates the export,
builds for the Pages base path, and uploads only `dist`. It uses official GitHub
actions pinned to their reviewed revisions. It installs no Python packages,
downloads no models, and starts no training. There are no scheduled runs.

Enable GitHub Pages with GitHub Actions as the publishing source, then run:

```sh
gh workflow run pages.yml --repo L2301/Angelos --ref main
gh run list --repo L2301/Angelos --workflow pages.yml --limit 1
```

GitHub Free does not provide Pages for private repositories. Do not silently
make a private source repository public: get an explicit visibility decision.
Alternatively, keep the source private and publish the generated files alone
to a separate public site repository. Use that repository name as the base path.
Do not use a paid-plan upgrade as an automatic workaround.

Local export, into a directory that does not already exist:

```sh
npm run build:site -- artifacts/site-release /Angelos/
```

Inspect `site-manifest.json`, then verify the published homepage, CSS/JS, image,
documentation and local-lab guide without authentication. Test both narrow and
desktop layouts under the real base path, and check that no `/api/` calls occur.
A successful push is not itself evidence that Pages deployed successfully.

References: [GitHub Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
