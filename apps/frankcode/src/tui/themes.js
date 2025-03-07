/**
 * Terminal UI Themes
 * 
 * Defines color themes for the TUI
 */

const chalk = require('chalk');

// Theme definitions
const themes = {
  dark: {
    bg: 'black',
    fg: 'white',
    border: {
      fg: 'blue',
      bg: 'black'
    },
    focus: {
      border: {
        fg: 'green',
        bg: 'black'
      }
    },
    label: {
      fg: 'white',
      bg: 'blue'
    },
    scrollbar: {
      bg: 'blue'
    }
  },
  light: {
    bg: 'white',
    fg: 'black',
    border: {
      fg: 'blue',
      bg: 'white'
    },
    focus: {
      border: {
        fg: 'green',
        bg: 'white'
      }
    },
    label: {
      fg: 'black',
      bg: 'cyan'
    },
    scrollbar: {
      bg: 'cyan'
    }
  },
  dracula: {
    bg: '#282a36',
    fg: '#f8f8f2',
    border: {
      fg: '#bd93f9',
      bg: '#282a36'
    },
    focus: {
      border: {
        fg: '#50fa7b',
        bg: '#282a36'
      }
    },
    label: {
      fg: '#f8f8f2',
      bg: '#6272a4'
    },
    scrollbar: {
      bg: '#bd93f9'
    }
  },
  solarized: {
    bg: '#002b36',
    fg: '#839496',
    border: {
      fg: '#2aa198',
      bg: '#002b36'
    },
    focus: {
      border: {
        fg: '#cb4b16',
        bg: '#002b36'
      }
    },
    label: {
      fg: '#fdf6e3',
      bg: '#073642'
    },
    scrollbar: {
      bg: '#586e75'
    }
  },
  nord: {
    bg: '#2e3440',
    fg: '#d8dee9',
    border: {
      fg: '#88c0d0',
      bg: '#2e3440'
    },
    focus: {
      border: {
        fg: '#a3be8c',
        bg: '#2e3440'
      }
    },
    label: {
      fg: '#eceff4',
      bg: '#3b4252'
    },
    scrollbar: {
      bg: '#5e81ac'
    }
  }
};

/**
 * Apply a theme to the screen
 * 
 * @param {Object} screen Blessed screen
 * @param {string} themeName Theme name
 */
function applyTheme(screen, themeName = 'dark') {
  // Get the theme
  const theme = themes[themeName] || themes.dark;
  
  // Apply to screen
  screen.style = {
    bg: theme.bg,
    fg: theme.fg
  };
  
  // Apply to all elements
  applyThemeToElements(screen, theme);
}

/**
 * Apply theme to all screen elements recursively
 * 
 * @param {Object} element Blessed element
 * @param {Object} theme Theme object
 */
function applyThemeToElements(element, theme) {
  // Set element styles
  if (element.style) {
    element.style.bg = theme.bg;
    element.style.fg = theme.fg;
    
    if (element.style.border) {
      element.style.border.fg = theme.border.fg;
      element.style.border.bg = theme.border.bg;
    }
    
    if (element.style.focus) {
      element.style.focus.border = {
        fg: theme.focus.border.fg,
        bg: theme.focus.border.bg
      };
    }
    
    if (element.style.scrollbar) {
      element.style.scrollbar.bg = theme.scrollbar.bg;
    }
    
    if (element.style.label) {
      element.style.label.fg = theme.label.fg;
      element.style.label.bg = theme.label.bg;
    }
  }
  
  // Apply to children
  if (element.children) {
    element.children.forEach(child => {
      applyThemeToElements(child, theme);
    });
  }
}

/**
 * Get available theme names
 * 
 * @returns {Array<string>} Theme names
 */
function getAvailableThemes() {
  return Object.keys(themes);
}

module.exports = {
  applyTheme,
  getAvailableThemes,
  themes
};