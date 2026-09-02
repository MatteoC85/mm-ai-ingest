# GitHub action — Phase 2, Commit 2

Branch: `refactor-phase1`

1. Extract the ZIP.
2. Copy the complete contents of the extracted folder over the current repository.
3. Choose **Replace files in the destination** when requested.
4. Do not delete the hidden `.git` folder.
5. In Source Control, confirm the intended changes include:
   - `main.py`;
   - `machinemind/infrastructure/openai_transport.py`;
   - Phase 2B documentation, tests and tools.
6. The following production files must not be modified by this commit:
   - `assistant_core_v2.py`;
   - `cloudbuild.yaml`;
   - `Dockerfile`;
   - `requirements.txt`;
   - `mm_promotion_gate.py`;
   - existing configuration and database modules.
7. Commit message:

   `refactor: extract OpenAI provider transport from production monolith`

8. Select **Commit and push**. The configured `refactor-phase1` trigger will build
   and deploy the commit automatically.
