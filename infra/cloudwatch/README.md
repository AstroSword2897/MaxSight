# CloudWatch examples

These are **templates** for operators. Replace `ACCOUNT`, `REGION`, `LogGroupName`, and SNS topic ARNs.

## Training job failure

Use **EventBridge** rule on `SageMaker Training Job State Change` with `Failure` status, target SNS, or use a metric filter on SageMaker log streams.

CLI sketch (alarm on failed training in account — adjust `MetricName` / dimensions to match your chosen metric):

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name maxsight-training-job-failed \
  --alarm-description "Alert when SageMaker training job fails" \
  --metric-name TrainingJobsFailed \
  --namespace AWS/SageMaker \
  --statistic Sum \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --dimensions Name=Host,Value=training \
  --treat-missing-data notBreaching
```

Prefer **EventBridge** for precise `SageMaker Training Job State Change` → `Failure` payloads; CloudWatch built-in SageMaker metrics vary by region and feature set.

Example event pattern JSON: **[eventbridge_training_job_failed.json](eventbridge_training_job_failed.json)** (`aws events put-rule --event-pattern file://...`). Point the rule target at SNS or a Lambda that opens a ticket.

## Endpoint 5xx

Create an alarm on **AWS/ApiGateway** or **SageMaker** invocation metrics if you front the endpoint with API Gateway, or use **CloudWatch Logs** metric filters on inference access logs.

## Log metric alignment (MaxSight training)

Training jobs emit lines parsed by `TRAINING_METRIC_DEFINITIONS`. After a successful short job, open the job log stream in CloudWatch Logs and confirm lines contain substrings like `Val mAP:` matching the regexes in `ml/infra/sagemaker_utils.py`.
