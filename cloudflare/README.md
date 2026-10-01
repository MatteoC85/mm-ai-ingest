# Worker Cloudflare del candidato

`worker.js` deriva dal sorgente di produzione `mm-ai-prod`, versione
`88ec46a7-7931-4733-9799-9b17b3c38ed5`, letto con Cloudflare MCP il 1 ottobre 2026.
La sola modifica candidata sceglie le etichette Root Cause `Why` / `Recommended checks`
oppure `Perché` / `Controlli consigliati` secondo la lingua della risposta.
Cache, accounting, autorizzazioni, timeout e payload restano quelli del sorgente letto.

SHA-256 candidato: `aaf5c8b78ac89a8032da3062044a6f7e89693212dfcd81559080faf8942926e6`.
La fixture `fixtures/worker.live-88ec46a7.js` è il sorgente live congelato
(SHA-256 `7ae430e0830003d3fe227acfb7035443fff261dfdbe14db6359113f656cb000a`),
necessario al confronto di neutralità dei metadati cache nei test qualità.

Il sorgente Worker ha un percorso di rilascio manuale, distinto dal trigger Cloud Build
del backend Cloud Run. Questa cartella non configura binding, secrets o deploy automatici;
i nomi di binding nel codice sono riferimenti runtime, senza valori delle credenziali.
I test usano esclusivamente credenziali sintetiche e servizi in memoria.

Con Node.js 20 o successivo, dalla radice del repository:

```sh
node cloudflare/verify-worker-quality.mjs
node cloudflare/verify-worker-controlled.mjs
```

Entrambi risolvono i file rispetto alla propria directory, bloccano destinazioni fetch
non previste e verificano l'hash del candidato senza riscriverlo. Non installano pacchetti
né eseguono richieste di rete. I report vengono scritti in `cloudflare/reports/`, ignorata
da Git. Sono 12 test qualità e 22 test di contratto controllati, separati dalle suite
storiche e senza attestazione del comportamento dei servizi live. Gli scenari Usage
`required` sono fixture offline; eseguirli non abilita alcuna configurazione.
