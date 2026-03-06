echo "⏳ Installing Python Libraries..."
# Updated list with ALL dependencies for RAG and Chat
pip install gradio langchain-ollama langchain-community langchain-core langchain-huggingface chromadb sentence-transformers opencv-python mediapipe numpy pandas joblib scikit-learn xgboost

echo "⏳ Installing GPU Detection Tools..."
apt-get update && apt-get install -y pciutils

echo "⏳ Installing zstd (required for Ollama)..."
apt-get install -y zstd

echo "⏳ Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "⏳ Starting Ollama Server..."
ollama serve &
sleep 5  # Wait for it to wake up

echo "⏳ Restoring 'medical' Model..."
cd /workspace/Chatbot
# Make sure we are in the right folder where Modelfile and qwen.gguf are
ollama create medical -f Modelfile

echo "✅ DONE! You can now run the app."