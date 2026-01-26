# TUI Guide for Proxima
### Complete Beginner's Guide to Using Proxima's Terminal User Interface

---

## Table of Contents

1. [Starting the TUI](#starting-the-tui)
2. [Understanding the Screen Layout](#understanding-the-screen-layout)
3. [Navigating Between Screens](#navigating-between-screens)
4. [Dashboard Screen](#dashboard-screen)
5. [Execution Monitor Screen](#execution-monitor-screen)
6. [Results Browser Screen](#results-browser-screen)
7. [Backend Management Screen](#backend-management-screen)
8. [Settings Screen](#settings-screen)
9. [Help Screen](#help-screen)
10. [Command Palette](#command-palette)
11. [Keyboard Shortcuts Reference](#keyboard-shortcuts-reference)
12. [Tips and Tricks](#tips-and-tricks)

---

## Starting the TUI

### How to Launch Proxima TUI

Open your terminal (Command Prompt or PowerShell on Windows) and type:

```bash
proxima ui
```

Press **Enter**. The TUI will open in full-screen mode showing a beautiful interface with colorful text and buttons.

**What you'll see:**
- A title bar at the top saying "Proxima - Quantum Simulation Orchestration"
- Navigation buttons at the bottom showing numbers 1-5
- The main Dashboard screen in the center

---

## Understanding the Screen Layout

Every screen in Proxima TUI has the same basic structure:

### Top Section (Header)
- **Title Bar**: Shows "Proxima" and the current screen name
- **Current Screen Name**: e.g., "Dashboard", "Execution Monitor", "Results Browser"

### Middle Section (Main Content)
- This is where all the information and buttons appear
- Different for each screen (Dashboard, Execution, Results, etc.)

### Bottom Section (Footer)
- **Navigation Bar**: Shows quick shortcuts
  - `[1] Dashboard` - Press 1 to go to Dashboard
  - `[2] Execution` - Press 2 to go to Execution Monitor
  - `[3] Results` - Press 3 to view Results
  - `[4] Backends` - Press 4 to manage Backends
  - `[5] Settings` - Press 5 to open Settings
  - `[Ctrl+P] Commands` - Opens the command search box
  - `[?] Help` - Press ? to see help
  - `[Ctrl+Q] Quit` - Exit the application

---

## Navigating Between Screens

### Three Ways to Navigate:

#### Method 1: Number Keys (Easiest)
Just press the number shown in brackets:
- Press **1** → Go to Dashboard
- Press **2** → Go to Execution Monitor
- Press **3** → Go to Results Browser
- Press **4** → Go to Backends
- Press **5** → Go to Settings

#### Method 2: Mouse Clicks
- Click on any button with your mouse
- Click on items in lists to select them

#### Method 3: Tab Navigation
- Press **Tab** to move to the next button or field
- Press **Shift+Tab** to move backward
- Press **Enter** to activate the selected button

---

## Dashboard Screen

**What is it?** Your home screen - shows an overview of everything.

### What You'll See:

#### 1. Welcome Section (Top)
```
Welcome to Proxima
Intelligent Quantum Simulation Orchestration
```
This just greets you when you enter.

#### 2. Quick Actions Buttons (Middle-Top)
Six clickable buttons for common tasks:

**[1] Run Simulation**
- Click this to start a new quantum simulation
- Opens the command palette to enter what you want to simulate

**[2] Compare Backends**
- Opens the Backends screen
- Lets you see which simulation engine is fastest

**[3] View Results**
- Opens the Results screen
- Shows results from your previous simulations

**[4] Manage Sessions**
- See all your work sessions
- Switch between different projects

**[5] Configure**
- Opens Settings screen
- Change preferences and options

**[?] Help**
- Opens the Help screen
- Shows detailed instructions

**How to use these buttons:**
- **With Mouse**: Just click on them
- **With Keyboard**: Press the number shown (1, 2, 3, etc.)

#### 3. Recent Sessions Table (Middle)
Shows a list of your last simulations:

| Column | What It Means |
|--------|---------------|
| **ID** | Unique code for the simulation (like a7b2c3d4) |
| **Task** | What you simulated (e.g., "Bell State", "GHZ 4-qubit") |
| **Backend** | Which simulator was used (Cirq, Qiskit, etc.) |
| **Status** | Whether it's done (✓ Done), running, or paused (⏸ Paused) |
| **Time** | How long ago it ran (e.g., "12s ago", "2m ago") |

**How to use the table:**
- **Arrow Keys**: Press ↑ (up) or ↓ (down) to select different rows
- **Enter**: Press Enter on a row to view that simulation's details
- **Mouse**: Click on any row to select it

#### 4. System Health Bar (Bottom)
Shows your computer's performance:
```
CPU: 23%  │  Memory: 52%  │  Backends: 3/6 healthy
```

- **CPU**: How much of your computer's brain is being used
- **Memory**: How much RAM is being used
- **Backends**: How many simulation engines are working properly

**What the numbers mean:**
- **Green** (0-70%): Everything is fine
- **Yellow** (70-85%): Getting busy
- **Red** (85-100%): Computer is working very hard

---

## Execution Monitor Screen

**What is it?** This screen shows a simulation while it's running - like watching a progress bar when downloading a file.

**How to get here:** Press **2** from anywhere in the TUI.

### What You'll See:

#### 1. Execution Info Panel (Top)
```
Execution Monitor
━━━━━━━━━━━━━━━━━
Backend: Cirq (StateVector)
Task: Bell State Simulation
Status: Running
```

Shows:
- **Backend**: Which simulator is being used
- **Task**: What's being simulated
- **Status**: Running, Paused, Completed, or Failed

#### 2. Progress Bar (Middle-Top)
```
Stage 2/4: Circuit Execution
[████████████████░░░░░░░░] 67%
Elapsed: 02:34  |  ETA: 01:15
```

**What you see:**
- **Stage Number**: Which step it's on (e.g., 2 out of 4)
- **Stage Name**: What's happening now (e.g., "Circuit Execution")
- **Progress Bar**: Visual bar showing completion percentage
  - **Filled blocks (████)**: Completed
  - **Empty blocks (░░░░)**: Not yet done
- **Elapsed**: Time already spent (02:34 = 2 minutes 34 seconds)
- **ETA**: Estimated time remaining (01:15 = 1 minute 15 seconds)

#### 3. Stage Timeline (Middle)
Shows all the steps in order:
```
1. ✓ Initialization      [Done]      10.2s
2. ▶ Circuit Execution   [Running]   45.8s
3. ○ Result Analysis     [Pending]   -
4. ○ Export              [Pending]   -
```

**Symbols explained:**
- **✓** (checkmark): Step is completed
- **▶** (play icon): Step is currently running
- **○** (circle): Step hasn't started yet

**How to use it:**
- Just watch the progress - no action needed
- Each step shows how long it took or is taking

#### 4. Control Buttons (Middle-Bottom)
Five buttons to control the running simulation:

**[P] Pause**
- **What it does**: Stops the simulation temporarily (you can resume it later)
- **When to use**: When you need to pause and come back later
- **Color**: Yellow/Orange (warning color)
- **How to use**: Click it OR press **P** key

**[R] Resume**
- **What it does**: Continues a paused simulation
- **When to use**: After you've paused a simulation
- **Color**: Green (success color)
- **How to use**: Click it OR press **R** key
- **Note**: Only works when something is paused (grayed out otherwise)

**[A] Abort**
- **What it does**: Stops and cancels the simulation completely
- **When to use**: When you want to stop and start over
- **Color**: Red (danger color)
- **How to use**: Click it OR press **A** key
- **Warning**: You'll lose the current simulation progress

**[Z] Rollback**
- **What it does**: Goes back to the last saved checkpoint
- **When to use**: If something went wrong and you want to try again from a save point
- **How to use**: Click it OR press **Z** key
- **Note**: Only works if you have a checkpoint saved (grayed out if no checkpoint exists)

**[L] Toggle Log**
- **What it does**: Shows or hides the detailed log messages
- **When to use**: To see technical details or save screen space
- **How to use**: Click it OR press **L** key

#### 5. Execution Log (Bottom)
```
[12:34:56] INFO  | Initializing backend...
[12:34:57] INFO  | Loading circuit definition...
[12:35:02] DEBUG | Applying gate H to qubit 0
[12:35:02] DEBUG | Applying CNOT to qubits 0,1
[12:35:05] INFO  | Executing with 1024 shots...
```

**What it shows:**
- **Timestamp**: Time of each message ([12:34:56] = 12:34 and 56 seconds)
- **Level**: 
  - **INFO**: General information (normal messages)
  - **DEBUG**: Technical details (for experts)
  - **WARNING**: Something to be careful about (yellow)
  - **ERROR**: Something went wrong (red)
- **Message**: What happened

**How to use:**
- **Scroll**: Use mouse wheel or arrow keys to scroll up/down
- **Hide/Show**: Press **L** to toggle visibility
- **Read**: Helps you understand what's happening behind the scenes

---

## Results Browser Screen

**What is it?** Shows the results from your completed simulations.

**How to get here:** Press **3** from anywhere in the TUI.

### What You'll See:

The screen is split into two parts (left and right):

#### Left Side: Results List
```
Results
━━━━━━━━━━━━━━━━━
▶ result_001.json
  result_002.json
  comparison_001.json
  bell_state_run.json
  ghz_4qubit.json
```

**What it shows:**
- List of all your saved simulation results
- **▶** (arrow): Shows which result is currently selected
- File names ending in `.json` (just a file format)

**How to use:**
- **Arrow Keys**: Press ↑ or ↓ to select different results
- **Mouse**: Click on any result to select it
- **Enter**: Press Enter to view details

#### Right Side: Result Details

**Header Section (Top)**
```
Simulation Results
━━━━━━━━━━━━━━━━━━━━━━━━━━
Backend: Cirq (StateVector)  │  Qubits: 4  │  Shots: 1024  │  Time: 45.2s
```

Shows:
- **Backend**: Which simulator was used
- **Qubits**: Number of quantum bits (how complex the simulation was)
- **Shots**: Number of times the experiment was repeated
- **Time**: How long it took to run

**Probability Distribution (Main Area)**

Shows the results in a bar chart format:
```
Measurement Outcomes
━━━━━━━━━━━━━━━━━━━━

|0000⟩  ████████████████████████ 512 (50.0%)
|1111⟩  ████████████████████████ 512 (50.0%)
|0001⟩  ░                          0 (0.0%)
|0010⟩  ░                          0 (0.0%)
```

**What you see:**
- **State (left)**: Quantum state like |0000⟩ or |1111⟩
  - Numbers show different measurement results
  - |0000⟩ means all qubits measured as 0
  - |1111⟩ means all qubits measured as 1
- **Bar**: Visual representation (more █ blocks = higher probability)
- **Count**: Number of times this outcome occurred (e.g., 512)
- **Percentage**: What % of total shots (e.g., 50.0%)

**In simple terms:** Think of it like rolling dice 1024 times and counting how many times you got each number.

**Action Buttons (Bottom)**

**View Full Stats**
- Shows complete statistical analysis
- Includes averages, standard deviation, etc.
- Click to open detailed statistics window

**Export JSON**
- Saves results as a JSON file
- Use when you want to open results in other programs
- Click to save file

**Export HTML**
- Saves results as a pretty HTML report
- Can open in web browser
- Good for sharing or printing
- Click to save file

**Compare**
- Compare this result with others
- See differences between simulations
- Click to open comparison view

---

## Backend Management Screen

**What is it?** Shows all available simulation engines and their status.

**How to get here:** Press **4** from anywhere in the TUI.

**What is a Backend?** Think of it like different car engines - each one runs your simulation in a different way. Some are fast, some are accurate, some work better for certain tasks.

### What You'll See:

#### Backend Cards Grid

The screen shows cards for 6 different backends:

**1. LRET Card**
```
┌──────────────────────────┐
│ ✓ LRET                   │
│ Local Realistic          │
│ Entanglement Theory      │
│                          │
│ ● Healthy                │
└──────────────────────────┘
```

**2. Cirq Card**
```
┌──────────────────────────┐
│ ✓ Cirq                   │
│ Google's quantum         │
│ framework                │
│                          │
│ ● Healthy                │
└──────────────────────────┘
```

**3. Qiskit Aer Card**
```
┌──────────────────────────┐
│ ✓ Qiskit Aer             │
│ IBM quantum              │
│ simulator                │
│                          │
│ ● Healthy                │
└──────────────────────────┘
```

**4. cuQuantum Card**
```
┌──────────────────────────┐
│ ✗ cuQuantum              │
│ NVIDIA GPU               │
│ acceleration             │
│                          │
│ ○ Unavailable            │
└──────────────────────────┘
```

**5. qsim Card**
```
┌──────────────────────────┐
│ ? qsim                   │
│ High-performance         │
│ simulator                │
│                          │
│ ○ Unknown                │
└──────────────────────────┘
```

**6. QuEST Card**
```
┌──────────────────────────┐
│ ? QuEST                  │
│ Quantum Exact            │
│ Simulation Toolkit       │
│                          │
│ ○ Unknown                │
└──────────────────────────┘
```

**Understanding the Symbols:**
- **✓** (Green checkmark): Backend is working and ready
- **✗** (Red X): Backend is not available (not installed or error)
- **?** (Gray question): Backend status unknown (needs health check)

**Understanding the Status:**
- **● Healthy** (Green): Ready to use
- **○ Unavailable** (Red): Cannot be used right now
- **○ Unknown** (Gray): Not tested yet

**How to select a backend:**
- **Mouse**: Click on any card
- **Tab**: Press Tab to move between cards
- **Enter**: Press Enter to select
- **Selected card**: Has a brighter border and highlighted background

#### Action Buttons (Bottom)

**Run Health Check** (Blue)
- Tests all backends to see if they're working
- Updates the status symbols and colors
- **When to use**: When you first start or after installing new backends
- **How to use**: Click it or press Enter when selected

**Compare Performance**
- Shows which backend is fastest for different tasks
- Displays speed comparison charts
- **When to use**: To choose the best backend for your simulation
- **How to use**: Click the button

**View Metrics**
- Shows detailed performance statistics
- Memory usage, CPU usage, speed benchmarks
- **When to use**: For advanced users who want detailed info
- **How to use**: Click the button

**Configure**
- Opens settings for backends
- Change timeout, memory limits, etc.
- **When to use**: To customize backend behavior
- **How to use**: Click the button

### Which Backend Should You Choose?

**For Beginners:**
- **LRET**: Fast and simple, great for learning
- **Cirq**: Good all-around choice, widely used

**For Speed:**
- **qsim**: Very fast for large simulations
- **cuQuantum**: Fastest if you have NVIDIA GPU

**For Accuracy:**
- **Qiskit Aer**: Industry standard, very reliable
- **QuEST**: Highly accurate for complex simulations

---

## Settings Screen

**What is it?** Where you configure and customize Proxima.

**How to get here:** Press **5** from anywhere in the TUI.

### Settings Sections:

#### 1. General Settings

**Default Backend**
```
Default Backend:          Cirq
```
- **What it is**: Which simulation engine to use automatically
- **How to change**: Click on the value to see options
- **Options**: LRET, Cirq, Qiskit Aer, Auto (chooses best)

**Default Shots**
```
Default Shots:           [1024    ]
```
- **What it is**: How many times to repeat each simulation
- **How to change**: Click in the box and type a number
- **Recommended**: 1024 (good balance), 10000 (more accurate but slower)

**Auto-save Results**
```
Auto-save Results:       [ON]  /  [ ]
```
- **What it is**: Automatically save results after each simulation
- **How to change**: Click the switch to toggle ON/OFF
- **[ON]**: Results saved automatically (recommended)
- **[ ]** (OFF): You must manually save

#### 2. LLM Configuration (AI Features)

**LLM** = Large Language Model (the AI that helps explain things)

**Provider**
```
Provider:                Ollama (Local)
```
- **What it is**: Which AI service to use
- **Options**:
  - **Ollama (Local)**: AI runs on your computer (free, private)
  - **OpenAI**: Uses ChatGPT (requires API key, costs money)
  - **Anthropic**: Uses Claude (requires API key, costs money)
  - **None**: Disable AI features

**Model**
```
Model:                   [llama2  ]
```
- **What it is**: Which specific AI model to use
- **How to change**: Type the model name
- **For Ollama**: llama2, llama3, mistral, etc.

**Enable Thinking**
```
Enable Thinking:         [ ]  /  [ON]
```
- **What it is**: Shows AI's thought process before answers
- **[ON]**: See how AI reasoned (slower but more transparent)
- **[ ]** (OFF): Just get the answer (faster)

#### 3. Display Settings

**Theme**
```
Theme:                   Dark
```
- **What it is**: Color scheme for the interface
- **Options**:
  - **Dark**: Black background, bright text (easier on eyes)
  - **Light**: White background, dark text (like paper)

**Compact Sidebar**
```
Compact Sidebar:         [ ]  /  [ON]
```
- **What it is**: Makes side panels smaller to show more content
- **[ON]**: Smaller sidebars, more space for main content
- **[ ]** (OFF): Normal size sidebars

**Show Log Panel**
```
Show Log Panel:          [ON]  /  [ ]
```
- **What it is**: Display technical log messages during execution
- **[ON]**: Show logs (recommended for troubleshooting)
- **[ ]** (OFF): Hide logs (cleaner interface)

#### Action Buttons (Bottom)

**Save Settings** (Blue, Primary Button)
- **What it does**: Saves all your changes
- **IMPORTANT**: Must click this or changes will be lost!
- **When to use**: After making any changes
- **Feedback**: Shows "Settings saved!" message

**Reset to Defaults**
- **What it does**: Changes everything back to original settings
- **When to use**: If you messed up and want to start over
- **Warning**: Erases all your custom settings

**Export Config**
- **What it does**: Saves settings to a file
- **When to use**: To backup your settings or share with others
- **Creates**: A config.json file you can save

**Import Config**
- **What it does**: Loads settings from a file
- **When to use**: To restore backup or use someone else's settings
- **How**: Opens file picker to select config.json file

---

## Help Screen

**What is it?** Built-in help and documentation.

**How to get here:** Press **?** from anywhere in the TUI.

### What You'll See:

#### Sections in Help Screen:

**1. Getting Started**
- Basic introduction to Proxima
- How to run your first simulation
- Common workflows

**2. Keyboard Shortcuts**
- Complete list of all keyboard commands
- Organized by category (Navigation, Execution, etc.)
- Quick reference table

**3. Screen Guides**
- Detailed explanation of each screen
- What each button does
- Tips for each screen

**4. Concepts**
- **What is a Backend?** - Explanation of simulation engines
- **What are Shots?** - Understanding repetitions
- **What are Qubits?** - Basic quantum computing concepts
- **Quantum States** - Understanding |0⟩, |1⟩, etc.

**5. Troubleshooting**
- Common problems and solutions
- Error message explanations
- When to ask for help

**6. External Resources**
- Links to full documentation
- GitHub repository
- Community support channels

**How to Navigate Help:**
- **Arrow Keys**: Scroll up and down
- **Page Up/Page Down**: Jump by full page
- **Home**: Go to top
- **End**: Go to bottom
- **Tab**: Jump between sections
- **Escape**: Close help and return to previous screen

---

## Command Palette

**What is it?** A searchable menu of all available commands - like a universal search box.

**How to open:** Press **Ctrl+P** from anywhere.

### What You'll See:

```
┌──────────────────────────────────────────────────────┐
│ Command Palette                                      │
├──────────────────────────────────────────────────────┤
│ Search: [run sim_______________________________]     │
├──────────────────────────────────────────────────────┤
│                                                      │
│ ▶ Run Simulation                      [Ctrl+R]      │
│   Start a new simulation run                        │
│                                                      │
│   Resume Execution                    [R]           │
│   Resume paused execution                           │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### How to Use:

**Step 1: Open the Palette**
- Press **Ctrl+P** (hold Ctrl, press P)
- A search box appears in the middle of screen

**Step 2: Type to Search**
- Start typing what you want to do
- Examples:
  - Type "run" → shows "Run Simulation"
  - Type "pause" → shows "Pause Execution"
  - Type "backend" → shows all backend-related commands
  - Type "export" → shows export options

**Step 3: Select Command**
- **Arrow Keys**: Press ↑ or ↓ to highlight different commands
- **Mouse**: Click on a command
- **Enter**: Press Enter to execute highlighted command

**Step 4: Command Executes**
- The palette closes
- Your selected command runs immediately

### Available Command Categories:

#### Execution Commands
- **Run Simulation** (Ctrl+R): Start new simulation
- **Pause Execution** (P): Pause running simulation
- **Resume Execution** (R): Continue paused simulation
- **Abort Execution** (A): Stop and cancel
- **Rollback** (Z): Return to checkpoint

#### Session Commands
- **New Session** (Ctrl+N): Create new work session
- **Switch Session**: Change to different session
- **Export Session**: Save session to file
- **View History**: See all past executions

#### Backend Commands
- **Switch Backend**: Choose different simulator
- **Health Check**: Test all backends
- **Compare Backends**: See performance comparison

#### LLM Commands
- **Configure LLM**: Set up AI features
- **Toggle Thinking**: Turn AI thinking on/off
- **Switch Provider**: Change AI service

#### Navigation Commands
- **Go to Dashboard** (1): Open dashboard
- **Go to Execution** (2): Open execution monitor
- **Go to Results** (3): Open results browser
- **Go to Backends** (4): Open backend management
- **Go to Settings** (5): Open settings
- **Show Help** (?): Open help screen

#### System Commands
- **Quit** (Ctrl+Q): Exit Proxima

### Tips for Command Palette:

**Fuzzy Search:**
- You don't need to type exact words
- Type "comp back" finds "Compare Backends"
- Type "rb" might find "Rollback"

**Learn Keyboard Shortcuts:**
- Each command shows its keyboard shortcut
- Commands you use often - remember their shortcuts
- Faster than opening palette every time

**Close Without Executing:**
- Press **Escape** to close without running anything
- Or click outside the palette box

---

## Keyboard Shortcuts Reference

### Essential Shortcuts (Must Know)

| Key | Action | Where It Works |
|-----|--------|----------------|
| **Ctrl+Q** | Quit Proxima | Everywhere |
| **Ctrl+P** | Open Command Palette | Everywhere |
| **?** | Show Help | Everywhere |
| **Escape** | Close/Cancel | Everywhere |
| **Tab** | Next field/button | Everywhere |
| **Enter** | Select/Activate | Everywhere |

### Navigation Shortcuts

| Key | Action | Description |
|-----|--------|-------------|
| **1** | Dashboard | Go to main screen |
| **2** | Execution | Go to execution monitor |
| **3** | Results | Go to results browser |
| **4** | Backends | Go to backend management |
| **5** | Settings | Go to configuration |

### Execution Control Shortcuts

| Key | Action | When To Use |
|-----|--------|-------------|
| **P** | Pause | During execution |
| **R** | Resume | When paused |
| **A** | Abort | Stop completely |
| **Z** | Rollback | Return to checkpoint |
| **L** | Toggle Log | Show/hide log panel |

### Dialog Shortcuts

| Key | Action | When To Use |
|-----|--------|-------------|
| **Enter** | Confirm | In any dialog |
| **Escape** | Cancel | In any dialog |
| **↑ ↓** | Navigate | In lists and menus |
| **Tab** | Switch category | In command palette |

### Permission Dialog Shortcuts

| Key | Action | Meaning |
|-----|--------|---------|
| **A** | Allow | Give permission once |
| **S** | Allow for Session | Allow for entire session |
| **D** | Deny | Refuse permission |
| **T** | Toggle Diff | Show/hide differences |

---

## Tips and Tricks

### For Absolute Beginners

**1. Start with Dashboard**
- Always begin at Dashboard (press 1)
- Use the Quick Action buttons - they're designed for beginners
- Don't worry about keyboard shortcuts at first - use mouse

**2. Watch an Execution**
- Go to Execution screen (press 2)
- See how simulations work in real-time
- Watch the progress bar and stage timeline
- This helps you understand what's happening

**3. Explore Results**
- Go to Results screen (press 3)
- Look at the probability bars
- They show what happened in the simulation
- Higher bars = that outcome happened more often

**4. Don't Touch These (Yet)**
- Rollback button - for advanced users
- Configure Backend - default settings are fine
- LLM Thinking mode - not needed when learning

### Speed Tips

**1. Use Number Keys**
- Pressing 1, 2, 3, 4, 5 is faster than clicking
- Much quicker than using mouse

**2. Learn These 3 Shortcuts**
- **Ctrl+P**: Command palette (find anything instantly)
- **?**: Help (when stuck)
- **Ctrl+Q**: Quit (when done)

**3. Command Palette is Your Friend**
- Can't find something? Press Ctrl+P and type it
- Faster than navigating through menus

### Understanding the Display

**Color Meanings:**
- **Green**: Good, healthy, success, completed
- **Yellow/Orange**: Warning, paused, needs attention
- **Red**: Error, failed, danger, abort
- **Blue**: Primary action, recommended choice
- **Gray**: Disabled, unavailable, unknown

**Progress Indicators:**
- **✓** (Checkmark): Done, completed, healthy
- **▶** (Play): Currently running, active
- **○** (Circle): Pending, not started, unknown
- **✗** (X): Failed, unavailable, error
- **⏸** (Pause): Paused, waiting
- **?** (Question): Unknown status

**Bars and Percentages:**
- **Progress bars**: Show completion (more filled = more done)
- **Probability bars**: Show likelihood (longer bar = happened more)
- **Health bars**: Show resource usage (higher = using more)

### Common Tasks Made Easy

**"I want to run a simulation"**
1. Press **1** (Dashboard)
2. Click **[1] Run Simulation** button
3. Command palette opens - type what you want
4. Press Enter

**"I want to check if it's done"**
1. Press **2** (Execution Monitor)
2. Look at the progress bar
3. 100% and all steps have ✓ = Done

**"I want to see my results"**
1. Press **3** (Results Browser)
2. Click on a result in the left list
3. View the probability distribution on the right

**"I want to change settings"**
1. Press **5** (Settings)
2. Click on any setting to change it
3. Click **Save Settings** button (IMPORTANT!)

**"I'm stuck or confused"**
1. Press **?** (Help)
2. Read the relevant section
3. Or press **Ctrl+Q** to quit and start over

**"I want to stop a running simulation"**
1. Press **2** (Execution Monitor)
2. Click **[A] Abort** button (or press A key)
3. Confirm if asked

**"I want to use a different backend"**
1. Press **4** (Backends)
2. Click on the backend card you want
3. It will be used for next simulation

### Troubleshooting

**Problem: Screen looks weird or glitchy**
- **Solution**: Resize your terminal window larger
- Proxima needs at least 80 characters wide, 24 lines tall

**Problem: Buttons don't respond to clicks**
- **Solution**: 
  - Try using keyboard shortcuts instead
  - Press Tab to select, Enter to activate
  - Restart Proxima (Ctrl+Q, then reopen)

**Problem: Can't see log messages**
- **Solution**: 
  - Press **2** to go to Execution
  - Press **L** to show log panel
  - Scroll down if needed

**Problem: Settings not saving**
- **Solution**: 
  - Press **5** to go to Settings
  - Make your changes
  - Click **Save Settings** button (don't forget this!)

**Problem: Don't know what a term means**
- **Solution**:
  - Press **?** for Help
  - Look in the "Concepts" section
  - Or check the glossary below

### Glossary for Beginners

**Backend**: A simulation engine - different ways to run quantum simulations (like different car engines)

**Qubit**: Quantum bit - the basic unit in quantum computing (like a regular computer bit but quantum)

**Shot**: One run of the simulation - simulations are repeated many times for accuracy

**State**: The condition of qubits - shown as |0⟩ (zero) or |1⟩ (one)

**Circuit**: The quantum program - a series of operations on qubits

**Execution**: Running the simulation - actually doing the computation

**Session**: Your work period - all the simulations you run in one sitting

**Checkpoint**: A save point - lets you rollback if something goes wrong

**LLM**: Large Language Model - the AI that helps explain things

**TUI**: Terminal User Interface - the text-based visual interface you're using

**Palette**: The command search box - quick access to all commands

**Health**: Status of backends - whether they're working or not

**ETA**: Estimated Time of Arrival - how long until completion

**Abort**: Stop immediately - cancels the current operation

**Rollback**: Go back - return to a previous save point

---

## Final Tips

### Remember:
✓ **You can't break anything** - Feel free to explore and click around  
✓ **Ctrl+Q always quits** - If stuck, just exit and restart  
✓ **? always shows help** - Press question mark when confused  
✓ **Numbers switch screens** - 1,2,3,4,5 for different screens  
✓ **Tab moves around** - Use Tab to navigate without mouse  

### Practice This Flow:
1. Open Proxima: `proxima ui`
2. Dashboard (1) → Click "Run Simulation"
3. Execution (2) → Watch progress
4. Results (3) → View outcomes
5. Help (?) → Learn more
6. Quit (Ctrl+Q) → Exit when done

### Get Comfortable By:
- Opening and closing the command palette (Ctrl+P, Escape)
- Switching between screens using number keys (1-5)
- Looking at the dashboard and understanding what you see
- Reading this guide section by section as you explore

**You're ready to use Proxima TUI!** Start with simple tasks and gradually explore more features. The interface is designed to guide you - just follow the buttons and prompts. Good luck!
