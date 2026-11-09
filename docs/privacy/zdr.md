# Zero Data Retention

Otari connects to multiple LLM providers to process inference requests. Each provider has its own policies regarding how request and response data are handled, stored, and used. Understanding these policies helps you select providers that meet your application's privacy, security, and compliance requirements.

Zero Data Retention (ZDR) means that a provider does not persist prompts or model-generated responses after processing a request. Data is processed temporarily in memory for as long as it takes to generate the response.

When evaluating a provider's data handling policies, there are three key aspects:

| Concept              | Description                                                                              |
| :------------------- | :--------------------------------------------------------------------------------------- |
| **Data retention**   | Whether prompts and responses are stored after processing, and for how long.             |
| **Model training**   | Whether request and response data can be used to train or improve models.                |
| **Metadata logging** | Whether operational data, such as timestamps, token usage, and request IDs, is retained. |

## **Provider data retention policies**

The following table summarizes the retention classifications of providers supported by Otari.

| Provider       | Data retention | Model training | Privacy Policy                                                        | Models |
| :------------- | :------------- | :------------- | :-------------------------------------------------------------------- | :----- |
| Amazon Bedrock | Zero retention | No             | [Privacy](https://www.together.ai/privacy)                            | 49     |
| Fireworks AI   | Zero retention | No             | [Privacy](https://fireworks.ai/privacy-policy)                        | 15     |
| Mistral AI     | Zero retention | No             | [Privacy](https://legal.mistral.ai/terms/get-started/#privacy-policy) | 25     |
| Nebius         | Zero retention | No             | [Privacy](https://docs.nebius.com/legal/privacy)                      | 22     |
| OVH Cloud      | Zero retention | No             | [Privacy](https://us.ovhcloud.com/legal/privacy-policy/)              | 10     |
| Together AI    | Zero retention | No             | [Privacy](https://www.together.ai/privacy)                            | 14     |

Regarding metadata logging, note that even when Zero Data Retention is enabled, providers may retain operational metadata for monitoring, security, and compliance. This data may include timestamps, token counts, request identifiers, and usage statistics, but should not include prompt or response content under the applicable ZDR policy.

For more information about the use of third parties, please refer to our [Privacy Policy](https://otari.ai/privacy).
