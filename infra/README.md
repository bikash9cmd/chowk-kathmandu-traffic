# Deploying to AWS

Everything is defined in [`template.yaml`](template.yaml) (AWS SAM). Region: `ap-south-1` (Mumbai) is the closest to Nepal.

## Prerequisites

- An AWS account and credentials (`aws configure`)
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) and Docker
- A trained model and features locally: `make evaluate && make train`

## Deploy

```bash
sam build --template-file infra/template.yaml
sam deploy --guided            # stack name e.g. chowk; set OpsEmail to get alarms and the budget warning
python scripts/seed_aws.py --bucket <DataBucketName from the stack outputs>
```

The seed script uploads the junction config, the training features and labels, and the local champion model, and points the SSM parameter at it.

## Add a camera

```bash
scripts/register_camera.sh maitighar-01
```

Creates the IoT thing, a certificate (files in `certs/maitighar-01/`, never commit them) and attaches the `chowk-camera` policy. The box publishes each 15-minute record to `$aws/rules/chowk_flows/chowk/flows/maitighar-01`, subscribes to `chowk/model/champion`, and gets temporary S3 credentials from the IoT credentials provider with role alias `chowk-camera`.

To test without a box, publish a record by hand:

```bash
aws iot-data publish --topic '$aws/rules/chowk_flows/chowk/flows/test' --cli-binary-format raw-in-base64-out \
  --payload '{"junction":"maitighar","approach":"from_thapathali","interval_start":"2026-10-21T08:00:00",
              "counts":{"two_wheeler":300,"light":80,"medium":6,"heavy":10},"pcu":269.0,"pcu_per_hour":1076.0,
              "minutes_observed":15,"model_version":"bootstrap"}'
curl "https://<ApiUrl>/junctions/maitighar/now"
```

## Roll back a model

```bash
aws s3 ls s3://<bucket>/models/
aws ssm put-parameter --name /chowk/champion-model-version --value <version> --overwrite
aws iot-data publish --topic chowk/model/champion --retain --cli-binary-format raw-in-base64-out \
  --payload '{"version":"<version>","key":"models/<version>/classifier.joblib"}'
```

## Tear down

```bash
sam delete
```

Empty the bucket first (including old versions; versioning is on), and delete any IoT things and certificates you registered.

## What's verified and what isn't

- `cfn-lint infra/template.yaml` passes (CI runs it).
- The four Lambda handlers run against mocked S3, DynamoDB and SSM in `tests/test_aws_handlers.py`, including validation, idempotent re-sends, the watchdog, the API and a retrain blocked by the gate.
- The stack has **not** been deployed to a real account, and no camera box has run the edge loop on a real junction. Expect small fixes on a first deploy (IoT policy variables, the arm64 image build).
