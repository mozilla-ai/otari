# Otari Gateway

Otari Gateway is an OpenAI-compatible API gateway that sits between your applications and LLM providers. It routes requests to 40+ providers through a single unified API, while giving you control over access, cost, and observability.

It can run standalone (you manage everything) or connected to [otari.ai](https://otari.ai) (provider routing, auth, and usage are handled for you).

## Documentation

- [Deployment](deployment.md) -- Get the gateway running with Docker.
- [Configuration](configuration.md) -- Config file reference and environment variables.
- [Modes](modes.md) -- Standalone vs connected to otari.ai.
- [API Reference](api-reference.md) -- All available endpoints.
- [Models](models.md) -- Supported providers and model format.
- [Merge Gateway Compatibility](merge-gateway-compatibility.md) -- How the OSS routing control plane maps to Merge Gateway-style concepts.
