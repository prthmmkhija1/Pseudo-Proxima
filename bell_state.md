# Proxima Agent Instructions

## Metadata
- name: Bell State Experiment
- version: 1.0.0
- author: Pratham

## Configuration
- backend: auto
- shots: 1000
- output_format: xlsx

## Tasks

### Task 1: Bell State Preparation
Create and measure a Bell state circuit.

**Type:** simulation
**Backend:** auto

```quantum
H 0
CNOT 0 1
MEASURE ALL
```

### Task 2: Analyze Results
Generate insights from the simulation results.

**Type:** analysis
**Use LLM:** optional

Analyze the measurement distribution and provide insights.

## Output
- Format: XLSX
- Include: fidelity, execution time, insights
