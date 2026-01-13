# Migration Guides

This section provides comprehensive guides for migrating between Proxima versions and backends.

## Version Migration

- [v0.1.x to v0.2.x](v0.1-to-v0.2.md) - Unified backend interface, enhanced plugin system
- [v0.2.x to v0.3.x](v0.2-to-v0.3.md) - Additional backends, Web API, hook system

## Backend Migration

- [Backend Migration Guide](backend-migration.md) - Migrating circuits between backends

## Quick Reference

### Version Compatibility Matrix

| Feature | v0.1 | v0.2 | v0.3 |
|---------|------|------|------|
| Cirq Backend | ✅ | ✅ | ✅ |
| Qiskit Aer Backend | ✅ | ✅ | ✅ |
| LRET Backend | ❌ | ✅ | ✅ |
| QuEST Backend | ❌ | ❌ | ✅ |
| cuQuantum Backend | ❌ | ❌ | ✅ |
| qsim Backend | ❌ | ❌ | ✅ |
| Plugin System | Basic | Enhanced | Hooks |
| Web API | ❌ | Basic | Full |
| Session Management | ❌ | ❌ | ✅ |
| TOML Config | ❌ | ✅ | ✅ |

### Upgrade Path

```
v0.1.x → v0.2.x → v0.3.x (current)
```

Always upgrade one major version at a time for the smoothest experience.

### Quick Upgrade Commands

```bash
# From v0.1 to v0.2
pip install proxima-agent>=0.2.0,<0.3.0
proxima migrate --from 0.1 --to 0.2

# From v0.2 to v0.3
pip install proxima-agent>=0.3.0
proxima migrate --from 0.2 --to 0.3
```

## Getting Help

If you encounter issues during migration:

1. Check the specific migration guide for your version
2. Review the [Troubleshooting](../troubleshooting.md) section
3. Search [GitHub Issues](https://github.com/yourusername/proxima/issues)
4. Ask on [Discord](https://discord.gg/proxima)
