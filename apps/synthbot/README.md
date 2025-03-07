# SynthBot

SynthBot is an open-source terminal-based coding agent powered by distributed large language models. It helps developers write, modify, and understand code directly from their terminal, with full project context awareness.

<p align="center">
  <img src="assets/synthbot-logo.png" alt="SynthBot Logo" width="200"/>
</p>

## Features

- **Terminal-Based UI**: Beautiful text-based interface that runs in your terminal.
- **Project-Aware**: Continuously scans your project files to maintain context.
- **Direct Code Modifications**: Makes changes to your files with your approval.
- **Context Management**: Monitors token usage and suggests session refreshes.
- **Git-Aware**: Understands git history and project structure.
- **Distributed LLM**: Leverages your local distributed LLM network for privacy and speed.

## Installation

```bash
# Install globally
npm install -g synthbot

# Or install locally
npm install synthbot
```

## Usage

Navigate to your project's root directory and run:

```bash
synthbot run
```

Or if installed locally:

```bash
npx synthbot run
```

### Command Options

```bash
# Start SynthBot with a specific configuration
synthbot run --config path/to/config.js

# Set the coordinator address for the distributed LLM
synthbot run --coordinator 192.168.1.100:5555

# Start with debug logging
synthbot run --verbose

# Show help
synthbot --help
```

## Keyboard Shortcuts

Within the TUI:

- `Ctrl+C`: Exit SynthBot
- `Ctrl+S`: Save conversation
- `Ctrl+R`: Refresh project context
- `Ctrl+F`: Find in files
- `Tab`: Switch focus between panels
- `Ctrl+L`: Clear current conversation
- `Ctrl+Z`: Undo last file modification

## Configuration

Create a `.synthbotrc.js` file in your home or project directory:

```javascript
module.exports = {
  // LLM connection settings
  llm: {
    coordinatorHost: "localhost",
    coordinatorPort: 5555,
    model: "llama-2-13b",
    temperature: 0.7
  },
  
  // Project context settings
  context: {
    maxTokens: 8192,
    excludeDirs: ["node_modules", "dist", ".git"],
    excludeFiles: ["*.lock", "*.log"],
    priorityFiles: ["README.md", "package.json"]
  },
  
  // UI preferences
  ui: {
    theme: "dark",
    layout: "default",
    fontSize: 14
  }
};
```

## Architecture

SynthBot is organized into the following main components:

- **Agent**: Core logic for the LLM interaction and code understanding.
- **API**: Communication with the distributed LLM network.
- **TUI**: Terminal user interface components.
- **Utils**: Helper utilities for filesystem, git, etc.

```
synthbot/
├── bin/           # Executable scripts
├── src/
│   ├── agent/     # LLM agent implementation
│   ├── api/       # API for LLM communication
│   ├── tui/       # Terminal UI components
│   └── utils/     # Utility functions
└── config/        # Configuration
```

## Integrating with DistributedLLM

SynthBot is designed to work with your local DistributedLLM setup. Make sure your coordinator is running:

```bash
# In your DistributedLLM directory
python src/main.py --mode coordinator
```

SynthBot will automatically connect to the coordinator and use the distributed LLM network for all inferences.

## Contributing

Contributions welcome! Please check out our [contributing guidelines](CONTRIBUTING.md).

## License

MIT