# Documentation and Paper Integrity Rule

Whenever you are asked to make changes to the codebase, add new features, or alter the system design, you MUST automatically perform the following documentation updates before concluding your task:

1. **Architecture (`docs/architecture.md`)**: Check if the changes affect the system architecture, data flow, or components. If they do, update `docs/architecture.md` to reflect these changes accurately.
2. **Changelog (`docs/CHANGELOG.md`)**: Always append a new entry to `docs/CHANGELOG.md` under the current date, summarizing the modifications made to the codebase or documentation.
3. **Methodology and Research Prompts (`docs/methodology.md`)**: Review `docs/methodology.md`. If the recent changes introduce new methodologies, metrics, or architectural concepts, update the relevant methods section and the LLM drafting prompts in its Appendix to include these new details.
4. **Main README Project Structure (`README.md`)**: Whenever new files, folders, scripts, datasets, model registry entries, benchmark artifacts, or major docs are added to the repository, review and update the `README.md` Project Structure section so the top-level repo map stays current and unambiguous.
