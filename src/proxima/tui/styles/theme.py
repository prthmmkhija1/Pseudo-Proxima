"""Proxima TUI Theme - Quantum-inspired dark theme with magenta accents.

A comprehensive color palette and styling system inspired by Crush AI.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import colorsys


@dataclass
class ProximaTheme:
    """Proxima TUI Theme - Quantum-inspired dark theme with magenta accents."""
    
    name: str = "proxima-dark"
    is_dark: bool = True
    
    # Primary Accent Colors (Magenta/Purple gradient)
    primary: str = "#FF00FF"          # Magenta
    primary_light: str = "#FF66FF"    # Light magenta
    primary_dark: str = "#AA00AA"     # Dark magenta
    secondary: str = "#AA00FF"        # Purple
    secondary_light: str = "#CC66FF"  # Light purple
    tertiary: str = "#6600CC"         # Deep purple
    accent: str = "#00FFFF"           # Cyan (quantum highlight)
    
    # Background Colors
    bg_darkest: str = "#0a0a0a"       # Deepest background
    bg_base: str = "#121212"          # Main background
    bg_base_lighter: str = "#1a1a1a"  # Slightly lighter
    bg_subtle: str = "#242424"        # Subtle panels
    bg_overlay: str = "#2a2a2a"       # Dialogs/modals
    bg_elevated: str = "#333333"      # Elevated elements
    
    # Foreground Colors
    fg_base: str = "#FFFFFF"          # Primary text
    fg_muted: str = "#B0B0B0"         # Secondary text
    fg_half_muted: str = "#909090"    # Dimmed text
    fg_subtle: str = "#707070"        # Very dim text
    fg_selected: str = "#FFFFFF"      # Selected text
    fg_disabled: str = "#505050"      # Disabled text
    
    # Border Colors
    border: str = "#333333"           # Default border
    border_focus: str = "#FF00FF"     # Focused border
    border_subtle: str = "#2a2a2a"    # Subtle border
    
    # Status Colors
    success: str = "#00FF66"          # Success green
    success_dark: str = "#00AA44"     # Dark success
    error: str = "#FF3333"            # Error red
    error_dark: str = "#AA2222"       # Dark error
    warning: str = "#FFAA00"          # Warning orange
    warning_dark: str = "#CC8800"     # Dark warning
    info: str = "#00AAFF"             # Info blue
    info_dark: str = "#0088CC"        # Dark info
    
    # Execution State Colors
    state_idle: str = "#808080"       # Gray
    state_planning: str = "#FFAA00"   # Orange
    state_ready: str = "#00AAFF"      # Blue
    state_running: str = "#00FF66"    # Green
    state_paused: str = "#FFAA00"     # Orange
    state_completed: str = "#00FF66"  # Green
    state_error: str = "#FF3333"      # Red
    state_aborted: str = "#FF6600"    # Orange-red
    state_recovering: str = "#AA00FF" # Purple
    
    # Memory Level Colors
    memory_ok: str = "#00FF66"        # Green (< 60%)
    memory_info: str = "#00AAFF"      # Blue (60-80%)
    memory_warning: str = "#FFAA00"   # Orange (80-95%)
    memory_critical: str = "#FF6600"  # Orange-red (95-98%)
    memory_abort: str = "#FF3333"     # Red (> 98%)
    
    # Backend Health Colors
    health_healthy: str = "#00FF66"   # Green
    health_degraded: str = "#FFAA00"  # Orange
    health_unhealthy: str = "#FF3333" # Red
    health_unknown: str = "#808080"   # Gray
    
    # Diff Colors (for comparisons)
    diff_insert_bg: str = "#1a2f1a"   # Green-tinted background
    diff_insert_fg: str = "#00FF66"   # Green text
    diff_delete_bg: str = "#2f1a1a"   # Red-tinted background
    diff_delete_fg: str = "#FF6666"   # Red text
    diff_change_bg: str = "#2a2a1a"   # Yellow-tinted background
    diff_change_fg: str = "#FFFF66"   # Yellow text
    
    # Quantum-specific Colors
    qubit_zero: str = "#00AAFF"       # |0⟩ state blue
    qubit_one: str = "#FF00FF"        # |1⟩ state magenta
    entangled: str = "#AA00FF"        # Entanglement purple
    superposition: str = "#00FFFF"    # Superposition cyan
    
    # Gradient Colors (for animations)
    gradient_start: str = "#AA00FF"   # Purple
    gradient_mid: str = "#FF00FF"     # Magenta
    gradient_end: str = "#FF66FF"     # Light magenta
    
    def get_execution_state_color(self, state: str) -> str:
        """Get color for an execution state."""
        state_colors = {
            "IDLE": self.state_idle,
            "PLANNING": self.state_planning,
            "READY": self.state_ready,
            "RUNNING": self.state_running,
            "PAUSED": self.state_paused,
            "COMPLETED": self.state_completed,
            "ERROR": self.state_error,
            "ABORTED": self.state_aborted,
            "RECOVERING": self.state_recovering,
        }
        return state_colors.get(state.upper(), self.fg_muted)
    
    def get_memory_level_color(self, level: str) -> str:
        """Get color for a memory level."""
        level_colors = {
            "OK": self.memory_ok,
            "INFO": self.memory_info,
            "WARNING": self.memory_warning,
            "CRITICAL": self.memory_critical,
            "ABORT": self.memory_abort,
        }
        return level_colors.get(level.upper(), self.fg_muted)
    
    def get_health_color(self, status: str) -> str:
        """Get color for backend health status."""
        health_colors = {
            "HEALTHY": self.health_healthy,
            "DEGRADED": self.health_degraded,
            "UNHEALTHY": self.health_unhealthy,
            "UNKNOWN": self.health_unknown,
        }
        return health_colors.get(status.upper(), self.health_unknown)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color."""
    return f"#{r:02x}{g:02x}{b:02x}"


def blend_colors(color1: str, color2: str, ratio: float = 0.5) -> str:
    """Blend two colors together.
    
    Args:
        color1: First hex color
        color2: Second hex color
        ratio: Blend ratio (0.0 = color1, 1.0 = color2)
    
    Returns:
        Blended hex color
    """
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    
    r = int(r1 + (r2 - r1) * ratio)
    g = int(g1 + (g2 - g1) * ratio)
    b = int(b1 + (b2 - b1) * ratio)
    
    return rgb_to_hex(r, g, b)


def create_gradient(colors: list[str], steps: int) -> list[str]:
    """Create a color gradient from multiple color stops.
    
    Args:
        colors: List of hex colors
        steps: Number of gradient steps
    
    Returns:
        List of hex colors forming the gradient
    """
    if len(colors) < 2:
        return colors * steps
    
    gradient = []
    segments = len(colors) - 1
    steps_per_segment = steps // segments
    
    for i in range(segments):
        for j in range(steps_per_segment):
            ratio = j / steps_per_segment
            gradient.append(blend_colors(colors[i], colors[i + 1], ratio))
    
    # Add final color
    gradient.append(colors[-1])
    
    return gradient[:steps]


def apply_gradient_to_text(text: str, colors: list[str]) -> str:
    """Apply a gradient to text using Rich markup.
    
    Args:
        text: The text to colorize
        colors: List of hex colors
    
    Returns:
        Text with Rich color markup
    """
    if not text or not colors:
        return text
    
    gradient = create_gradient(colors, len(text))
    result = []
    
    for i, char in enumerate(text):
        if char.isspace():
            result.append(char)
        else:
            color = gradient[min(i, len(gradient) - 1)]
            result.append(f"[{color}]{char}[/]")
    
    return "".join(result)


# ============== THEME DEFINITIONS ==============

# Theme 1: Default Proxima Dark (Quantum-inspired magenta)
PROXIMA_DARK = ProximaTheme()

# Theme 2: Ocean Deep - Deep blue oceanic theme
OCEAN_DEEP = ProximaTheme(
    name="ocean-deep",
    is_dark=True,
    primary="#0088FF",
    primary_light="#4AA8FF",
    primary_dark="#0055AA",
    secondary="#00AACC",
    secondary_light="#44CCEE",
    tertiary="#004488",
    accent="#00FFCC",
    bg_darkest="#040810",
    bg_base="#0a1520",
    bg_base_lighter="#101a28",
    bg_subtle="#182838",
    bg_overlay="#203040",
    bg_elevated="#284050",
    fg_base="#E0F0FF",
    fg_muted="#A0C0E0",
    border="#203858",
    border_focus="#0088FF",
    success="#00FF88",
    error="#FF4466",
    warning="#FFAA44",
    info="#00CCFF",
    qubit_zero="#00CCFF",
    qubit_one="#0066FF",
    entangled="#4488FF",
    superposition="#00FFCC",
    gradient_start="#004488",
    gradient_mid="#0088FF",
    gradient_end="#00CCFF",
)

# Theme 3: Forest Night - Nature-inspired green theme
FOREST_NIGHT = ProximaTheme(
    name="forest-night",
    is_dark=True,
    primary="#00CC66",
    primary_light="#44EE88",
    primary_dark="#008844",
    secondary="#00AA55",
    secondary_light="#44CC77",
    tertiary="#006633",
    accent="#88FF44",
    bg_darkest="#040804",
    bg_base="#0a120a",
    bg_base_lighter="#101810",
    bg_subtle="#182818",
    bg_overlay="#203820",
    bg_elevated="#284828",
    fg_base="#E0FFE8",
    fg_muted="#A0D8B0",
    border="#1a3820",
    border_focus="#00CC66",
    success="#44FF88",
    error="#FF6644",
    warning="#FFCC44",
    info="#44CCFF",
    qubit_zero="#44DDAA",
    qubit_one="#00CC66",
    entangled="#00AA88",
    superposition="#88FF44",
    gradient_start="#006633",
    gradient_mid="#00CC66",
    gradient_end="#44EE88",
)

# Theme 4: Sunset Glow - Warm orange/red theme
SUNSET_GLOW = ProximaTheme(
    name="sunset-glow",
    is_dark=True,
    primary="#FF6633",
    primary_light="#FF8855",
    primary_dark="#CC4422",
    secondary="#FF8800",
    secondary_light="#FFAA44",
    tertiary="#CC5500",
    accent="#FFCC00",
    bg_darkest="#100804",
    bg_base="#180c08",
    bg_base_lighter="#201410",
    bg_subtle="#2a1c18",
    bg_overlay="#382820",
    bg_elevated="#483828",
    fg_base="#FFF8F0",
    fg_muted="#E0C8B8",
    border="#3a2820",
    border_focus="#FF6633",
    success="#88FF44",
    error="#FF4444",
    warning="#FFCC00",
    info="#44AAFF",
    qubit_zero="#FFAA44",
    qubit_one="#FF6633",
    entangled="#FF4400",
    superposition="#FFCC00",
    gradient_start="#CC4422",
    gradient_mid="#FF6633",
    gradient_end="#FFAA44",
)

# Theme 5: Arctic Ice - Cool white/blue theme
ARCTIC_ICE = ProximaTheme(
    name="arctic-ice",
    is_dark=True,
    primary="#88CCFF",
    primary_light="#AADDFF",
    primary_dark="#5599CC",
    secondary="#66AADD",
    secondary_light="#88CCEE",
    tertiary="#4488BB",
    accent="#CCFFFF",
    bg_darkest="#080c10",
    bg_base="#0c1218",
    bg_base_lighter="#121a22",
    bg_subtle="#1a242e",
    bg_overlay="#223040",
    bg_elevated="#2a3848",
    fg_base="#F8FCFF",
    fg_muted="#C8DCE8",
    border="#2a3848",
    border_focus="#88CCFF",
    success="#88FFAA",
    error="#FF6688",
    warning="#FFCC88",
    info="#88DDFF",
    qubit_zero="#AADDFF",
    qubit_one="#6699CC",
    entangled="#88BBEE",
    superposition="#CCFFFF",
    gradient_start="#4488BB",
    gradient_mid="#88CCFF",
    gradient_end="#CCFFFF",
)

# Theme 6: Neon Nights - Cyberpunk neon theme
NEON_NIGHTS = ProximaTheme(
    name="neon-nights",
    is_dark=True,
    primary="#FF00FF",
    primary_light="#FF66FF",
    primary_dark="#AA00AA",
    secondary="#00FFFF",
    secondary_light="#66FFFF",
    tertiary="#FF0088",
    accent="#00FF88",
    bg_darkest="#08000c",
    bg_base="#0c0014",
    bg_base_lighter="#140020",
    bg_subtle="#1c002c",
    bg_overlay="#280040",
    bg_elevated="#340050",
    fg_base="#FFFFFF",
    fg_muted="#D0B0D8",
    border="#40006a",
    border_focus="#FF00FF",
    success="#00FF88",
    error="#FF0044",
    warning="#FF8800",
    info="#00CCFF",
    qubit_zero="#00FFFF",
    qubit_one="#FF00FF",
    entangled="#FF0088",
    superposition="#00FF88",
    gradient_start="#FF0088",
    gradient_mid="#FF00FF",
    gradient_end="#00FFFF",
)

# Theme 7: Midnight Rose - Elegant pink/purple theme
MIDNIGHT_ROSE = ProximaTheme(
    name="midnight-rose",
    is_dark=True,
    primary="#FF6699",
    primary_light="#FF88AA",
    primary_dark="#CC4477",
    secondary="#CC4488",
    secondary_light="#DD6699",
    tertiary="#AA3366",
    accent="#FFAACC",
    bg_darkest="#0c0408",
    bg_base="#120810",
    bg_base_lighter="#1a0c18",
    bg_subtle="#241420",
    bg_overlay="#301c2a",
    bg_elevated="#3c2838",
    fg_base="#FFF0F8",
    fg_muted="#E0C0D0",
    border="#3a2030",
    border_focus="#FF6699",
    success="#88FF88",
    error="#FF4466",
    warning="#FFAA66",
    info="#88AAFF",
    qubit_zero="#FF88AA",
    qubit_one="#DD4488",
    entangled="#CC4488",
    superposition="#FFAACC",
    gradient_start="#AA3366",
    gradient_mid="#FF6699",
    gradient_end="#FFAACC",
)

# Theme 8: Golden Hour - Luxurious gold theme
GOLDEN_HOUR = ProximaTheme(
    name="golden-hour",
    is_dark=True,
    primary="#FFD700",
    primary_light="#FFE44D",
    primary_dark="#CC9900",
    secondary="#FFA500",
    secondary_light="#FFBB33",
    tertiary="#CC7700",
    accent="#FFFFAA",
    bg_darkest="#100c04",
    bg_base="#181208",
    bg_base_lighter="#201810",
    bg_subtle="#2a2018",
    bg_overlay="#382820",
    bg_elevated="#48382a",
    fg_base="#FFFFF0",
    fg_muted="#E8D8C0",
    border="#3a2818",
    border_focus="#FFD700",
    success="#88FF66",
    error="#FF5544",
    warning="#FFAA33",
    info="#55AAFF",
    qubit_zero="#FFE44D",
    qubit_one="#FFAA00",
    entangled="#FFD700",
    superposition="#FFFFAA",
    gradient_start="#CC7700",
    gradient_mid="#FFD700",
    gradient_end="#FFFFAA",
)

# Theme 9: Emerald City - Rich green theme
EMERALD_CITY = ProximaTheme(
    name="emerald-city",
    is_dark=True,
    primary="#50C878",
    primary_light="#70E898",
    primary_dark="#308858",
    secondary="#40A060",
    secondary_light="#60C080",
    tertiary="#207040",
    accent="#88FFAA",
    bg_darkest="#040a06",
    bg_base="#08120c",
    bg_base_lighter="#0c1a12",
    bg_subtle="#142418",
    bg_overlay="#1c3020",
    bg_elevated="#244028",
    fg_base="#F0FFF8",
    fg_muted="#C0E0D0",
    border="#1c3020",
    border_focus="#50C878",
    success="#88FF88",
    error="#FF6666",
    warning="#FFCC55",
    info="#55BBFF",
    qubit_zero="#70E898",
    qubit_one="#40A060",
    entangled="#50C878",
    superposition="#88FFAA",
    gradient_start="#207040",
    gradient_mid="#50C878",
    gradient_end="#88FFAA",
)

# Theme 10: Violet Dreams - Deep purple theme
VIOLET_DREAMS = ProximaTheme(
    name="violet-dreams",
    is_dark=True,
    primary="#9966FF",
    primary_light="#BB88FF",
    primary_dark="#6633CC",
    secondary="#7744DD",
    secondary_light="#9966EE",
    tertiary="#5522AA",
    accent="#CCAAFF",
    bg_darkest="#080410",
    bg_base="#0c0818",
    bg_base_lighter="#140c20",
    bg_subtle="#1c1430",
    bg_overlay="#281c40",
    bg_elevated="#342850",
    fg_base="#F8F0FF",
    fg_muted="#D0C0E8",
    border="#2a1840",
    border_focus="#9966FF",
    success="#88FF88",
    error="#FF5577",
    warning="#FFAA66",
    info="#66AAFF",
    qubit_zero="#BB88FF",
    qubit_one="#7744DD",
    entangled="#9966FF",
    superposition="#CCAAFF",
    gradient_start="#5522AA",
    gradient_mid="#9966FF",
    gradient_end="#CCAAFF",
)

# Theme 11: Light Mode (Bonus) - Clean light theme
LIGHT_MODE = ProximaTheme(
    name="light-mode",
    is_dark=False,
    primary="#6633CC",
    primary_light="#8855EE",
    primary_dark="#4422AA",
    secondary="#5522BB",
    secondary_light="#7744DD",
    tertiary="#3311AA",
    accent="#00AAFF",
    bg_darkest="#FFFFFF",
    bg_base="#F8F8FA",
    bg_base_lighter="#F0F0F5",
    bg_subtle="#E8E8F0",
    bg_overlay="#E0E0E8",
    bg_elevated="#D8D8E0",
    fg_base="#1a1a2e",
    fg_muted="#404060",
    fg_half_muted="#606080",
    fg_subtle="#8080A0",
    fg_selected="#1a1a2e",
    fg_disabled="#B0B0C0",
    border="#D0D0E0",
    border_focus="#6633CC",
    border_subtle="#E0E0E8",
    success="#00AA44",
    success_dark="#008833",
    error="#DD2222",
    error_dark="#AA1111",
    warning="#DD8800",
    warning_dark="#AA6600",
    info="#0088DD",
    info_dark="#0066AA",
    qubit_zero="#0088DD",
    qubit_one="#6633CC",
    entangled="#5522BB",
    superposition="#00AAFF",
    gradient_start="#6633CC",
    gradient_mid="#8855EE",
    gradient_end="#AA88FF",
)

# All available themes
THEMES = {
    "proxima-dark": PROXIMA_DARK,
    "ocean-deep": OCEAN_DEEP,
    "forest-night": FOREST_NIGHT,
    "sunset-glow": SUNSET_GLOW,
    "arctic-ice": ARCTIC_ICE,
    "neon-nights": NEON_NIGHTS,
    "midnight-rose": MIDNIGHT_ROSE,
    "golden-hour": GOLDEN_HOUR,
    "emerald-city": EMERALD_CITY,
    "violet-dreams": VIOLET_DREAMS,
    "light-mode": LIGHT_MODE,
}


# Global theme instance
_current_theme: ProximaTheme = ProximaTheme()


def get_theme() -> ProximaTheme:
    """Get the current theme."""
    return _current_theme


def set_theme(theme: ProximaTheme) -> None:
    """Set the current theme."""
    global _current_theme
    _current_theme = theme


def set_theme_by_name(name: str) -> bool:
    """Set the current theme by name.
    
    Args:
        name: Theme name (e.g., 'ocean-deep', 'neon-nights')
    
    Returns:
        True if theme was found and set, False otherwise
    """
    global _current_theme
    if name in THEMES:
        _current_theme = THEMES[name]
        return True
    return False


def get_theme_names() -> list[str]:
    """Get list of all available theme names."""
    return list(THEMES.keys())


def get_theme_by_name(name: str) -> ProximaTheme | None:
    """Get a theme by name without setting it as current."""
    return THEMES.get(name)
