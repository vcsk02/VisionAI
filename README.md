# VisionAI

VisionAI is a modern web client for experimenting with and interfacing to computer vision models. It’s built with Vite and TypeScript, providing a fast, type-safe development experience that’s easy to extend.

Demo Video: https://drive.google.com/file/d/1hIhXnMm4Ti3B2IGSRLTjcrrqqe6poLdZ/view?usp=sharing

Presentation Slides: https://docs.google.com/presentation/d/1IYoV1sVOmkXa1VnpouBnPfH5ukXI2xkx/edit?usp=sharing&ouid=117089919734072744686&rtpof=true&sd=true

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running in Development](#running-in-development)
  - [Building for Production](#building-for-production)
- [Configuration](#configuration)
- [Available Scripts](#available-scripts)
- [Development Notes](#development-notes)
- [Contributing](#contributing)
- [License](#license)
- [Methodology](#methodology)
- [Evaluation](#evaluation)

---

## Features

- Model-agnostic UI – Built to plug into different CV/AI backends via HTTP APIs.
- Image-first workflow – Designed around uploading or selecting images to send to models.
- Structured results – Layout geared toward showing predictions, bounding boxes, or other model output.
- Extensible – Add new “model cards” or views by dropping new components into `src/`.
- Fast dev loop – Vite-powered HMR (hot module replacement) for rapid iteration.

---

## Tech Stack

**Build tool:** Vite  
**Language:** TypeScript  

**Bundled tooling:**

- ESLint (`eslint.config.js`)
- TypeScript configs (`tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json`)

---

## Project Structure

Based on the repository layout:

```text
VisionAI/
├── public/               # Static assets (favicon, images, etc.)
├── src/                  # Source code (components, hooks, utilities)
├── index.html            # HTML entry point
├── package.json          # Project metadata, dependencies & scripts
├── vite.config.ts        # Vite configuration
├── tsconfig.json         # Base TypeScript config
├── tsconfig.app.json     # App-specific TypeScript config
├── tsconfig.node.json    # Node-specific TypeScript config
````

---

## Getting Started

### Prerequisites

* Node.js ≥ 18 (recommended)
* npm ≥ 9

Check your versions:

```bash
node -v
npm -v
```

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/vcsk02/VisionAI.git
cd VisionAI
npm install
```

### Running in Development

Start the local development server with hot-module replacement:

```bash
npm run dev
```

Then open the URL printed in the terminal (typically `http://localhost:5173`).

### Building for Production

Create an optimized production build:

```bash
npm run build
```

To preview the production build locally (if configured in `package.json`):

```bash
npm run preview
```

---

## Configuration

Most configuration lives in three places:

* **Vite** – `vite.config.ts`: Aliases, dev server configuration, build options, etc.
* **TypeScript** – `tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json`: Compiler options, path aliases and strictness settings.
* **Environment variables** – Typically via `.env` files (recommended).

Example environment variable configuration:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

Then inside your code, use `import.meta.env.VITE_API_BASE_URL` to call your backend.

---

## Available Scripts

* `npm run dev` – Start the development server.
* `npm run build` – Type-check (if configured) and create a production build.
* `npm run preview` – Preview the production build locally.
* `npm run lint` – Run ESLint on the codebase (if configured).

---

## Development Notes

* Keep components and hooks inside `src/` small and composable.
* If you wire this up to multiple vision models, consider a common interface (e.g. `VisionModelClient`) for different backends.
* Use a central place for API calls (e.g. `src/api/` or `src/services/`).
* Document any custom model outputs (bounding boxes, masks, captions) so contributors know how to render them.

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository.

2. Create a feature branch:

   ```bash
   git checkout -b feature/my-awesome-feature
   ```

3. Commit your changes with clear messages.

4. Push the branch and open a Pull Request.

Please keep PRs focused and small where possible.

## Methodology

We follow the CRISP-DM process for this project (see [docs/CRISP_DM.md](docs/CRISP_DM.md)).

## Evaluation

Model metrics and plots are stored in `artifacts/metrics/`.  
For details on how we split the data, which metrics we use, and how to interpret the plots, see [docs/evaluation.md](docs/evaluation.md).
