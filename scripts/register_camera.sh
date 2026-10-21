#!/usr/bin/env bash
# Register one camera box as an IoT thing with its own certificate.
#   scripts/register_camera.sh maitighar-01
# Copy the three files in ./certs/<thing>/ to the box. They are secrets: never commit them.
set -euo pipefail
THING="$1"
OUT="certs/${THING}"
mkdir -p "${OUT}"
aws iot create-thing --thing-name "${THING}" >/dev/null
ARN=$(aws iot create-keys-and-certificate --set-as-active \
  --certificate-pem-outfile "${OUT}/cert.pem" --private-key-outfile "${OUT}/private.key" \
  --query certificateArn --output text)
aws iot attach-policy --policy-name chowk-camera --target "${ARN}"
aws iot attach-thing-principal --thing-name "${THING}" --principal "${ARN}"
curl -s -o "${OUT}/AmazonRootCA1.pem" https://www.amazontrust.com/repository/AmazonRootCA1.pem
echo "endpoint: $(aws iot describe-endpoint --endpoint-type iot:Data-ATS --query endpointAddress --output text)"
echo "credentials endpoint: $(aws iot describe-endpoint --endpoint-type iot:CredentialProvider --query endpointAddress --output text)"
echo "publish to: \$aws/rules/chowk_flows/chowk/flows/${THING}"
