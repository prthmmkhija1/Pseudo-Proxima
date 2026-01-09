# Adding Backends

## Steps
1. Implement the backend adapter interface (execute, connect, disconnect).
2. Provide capability metadata (supports shots, max qubits, noise model).
3. Register the backend in the config and backend registry.
4. Add tests: unit for adapter, integration for planner interaction.
5. Update docs and examples.

## Adapter checklist
- Input validation and timeout handling
- Consistent result schema: counts, duration, metadata
- Error propagation with clear messages

## Example skeleton
```python
class MyBackendAdapter(BaseBackendAdapter):
    def execute(self, circuit, options):
        ...
```
