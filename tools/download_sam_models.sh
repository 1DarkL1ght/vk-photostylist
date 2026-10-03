DEFAULT_MODELS=(
    "mobilesam_fp32"
    "mobilesam_int8"
    "sam_b_fp32"
    "sam_b_int8"
    "sam_l_fp32"
    "sam_l_int8"
    "sam_h_fp32"
    "sam_h_int8"
)

declare -A REPO_URLS
REPO_URLS=(
    ["mobilesam_fp32"]="huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/mobile_sam_20230629.zip"
    ["mobilesam_int8"]="huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/mobile_sam_20230629_quant.zip"
    ["sam_b_fp32"]="huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_b_01ec64.zip"
    ["sam_b_int8"]="huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_b_01ec64_quant.zip"
    ["sam_l_fp32"]="huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_l_0b3195.zip"
    ["sam_l_int8"]="huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_l_0b3195_quant.zip"
    ["sam_h_fp32"]="huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_h_4b8939.zip"
    ["sam_h_int8"]="huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/sam_vit_h_01ec64_quant.zip"
)

TARGET_DIR="models"

if [ -n "$1" ]; then
    if [ -z "${REPO_URLS[$1]}" ]; then
        echo "Error: Model '$1' not found in the list of available."
        echo "Available models are: ${!REPO_URLS[@]}."
        exit 1
    fi
    MODELS_TO_DOWNLOAD=("$1")
else
    MODELS_TO_DOWNLOAD=("${DEFAULT_MODELS[@]}")
fi


for MODEL in "${MODELS_TO_DOWNLOAD[@]}"; do
    URL="${REPO_URLS[$MODEL]}"
    TARGET_DIR="./models/$MODEL"
    ZIP_URL="${URL}/resolve/main/?download=true"
    ZIP_FILE="temp_${MODEL}.zip"

    echo " Downloading: $MODEL"
    mkdir -p "$TARGET_DIR"
    curl -L -o "$ZIP_FILE" "$ZIP_URL"

    if [ ! -f "$ZIP_FILE" ] || [ ! -s "$ZIP_FILE" ]; then
        echo "Error while downloading $MODEL."
        rm -f "$ZIP_FILE"
        continue
    fi

    echo "-> Inflating..."
    unzip -o "$ZIP_FILE" -d "$TARGET_DIR"
    rm "$ZIP_FILE"
done
