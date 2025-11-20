# app.py - Fixed Version
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import time
import json

# Set page config
st.set_page_config(
    page_title="🚀 SPoC: Pseudocode to Code Converter",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned LoRA model and tokenizer"""
    try:
        # FIXED: Correct model path
        model_path = "spoc_gpt2_finetuned"
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model directory not found: {model_path}")
            st.info("💡 Make sure the model directory is in the same folder as app.py")
            return None, None, None
        
        st.info("🔄 Loading model... This might take a minute.")
        
        # Check adapter config to get base model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', 'gpt2')
        else:
            base_model_name = 'gpt2'
        
        st.write(f"📋 Base model: {base_model_name}")
        
        # FIXED: Use AutoTokenizer and AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        st.success(f"✅ Model loaded successfully on {device}!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)
        return None, None, None

def generate_code(model, tokenizer, device, pseudocode, max_length=512, temperature=0.7, num_beams=5):
    """Generate code from pseudocode"""
    try:
        # FIXED: Use the correct prompt format matching training data
        prompt = f"🧩 PSEUDOCODE:\n\n{pseudocode}\n\n💻 CODE:\n\n"
        
        # Tokenize input
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,  # Leave room for generation
            padding=True
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Generate code
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                temperature=temperature,
                num_beams=num_beams,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=1.2,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the code part after the prompt
        if "💻 CODE:" in generated_text:
            code_part = generated_text.split("💻 CODE:")[-1].strip()
        else:
            # Fallback: remove the prompt
            code_part = generated_text[len(prompt):].strip()
        
        return code_part
        
    except Exception as e:
        return f"Error during generation: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 SPoC: Pseudocode to Code Converter</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; margin-bottom: 2rem;'>
        Transform natural language pseudocode into executable C++ code using AI! 
        <br>Powered by fine-tuned GPT-2 with LoRA
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("---")
        st.subheader("📊 Model Information")
        st.info("""
        **Model:** GPT-2 with LoRA adapters  
        **Training:** SPoC dataset  
        **Task:** Pseudocode → C++ Code  
        **Parameters:** ~811K trainable
        """)
        
        st.markdown("---")
        st.subheader("🎛️ Generation Parameters")
        
        max_length = st.slider(
            "Max New Tokens", 
            min_value=128, 
            max_value=768, 
            value=512, 
            step=64,
            help="Maximum number of tokens to generate"
        )
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.1, 
            max_value=1.5, 
            value=0.7, 
            step=0.1,
            help="Higher = more creative, Lower = more deterministic"
        )
        
        num_beams = st.slider(
            "Beam Search", 
            min_value=1, 
            max_value=10, 
            value=5, 
            step=1,
            help="Number of beams for beam search (higher = better quality)"
        )
        
        st.markdown("---")
        st.subheader("💡 Tips for Best Results")
        st.markdown("""
        - ✅ Be specific and clear
        - ✅ Use simple variable names
        - ✅ Describe logic step-by-step
        - ✅ One operation per line
        - ✅ Use standard programming terms
        """)
        
        st.markdown("---")
        st.subheader("📚 Supported Patterns")
        st.markdown("""
        - Variables & I/O
        - Conditionals (if/else)
        - Loops (for/while)
        - Arrays & Strings
        - Functions
        - Basic Math operations
        """)
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None:
        st.error("❌ Failed to load the model. Please check the following:")
        
        # FIXED: Show correct model path
        model_path = "spoc_gpt2_finetuned"
        
        if os.path.exists(model_path):
            st.write("📁 **Files found in model directory:**")
            files = os.listdir(model_path)
            for file in files:
                st.write(f"  ✓ {file}")
            
            st.info("💡 Model files found but failed to load. Check the error message above.")
        else:
            st.error(f"📁 Model directory not found: `{model_path}`")
            st.info("""
            **To fix this:**
            1. Make sure you've trained the model
            2. Place the `spoc_gpt2_finetuned` folder in the same directory as `app.py`
            3. Restart the Streamlit app
            """)
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📝 Input: Pseudocode")
        
        # Example pseudocode matching training format
        example_pseudocode = """let s = array of characters of length 50
read s
let res be a string
initialize integer n value as strlen(s)
assign the integer value start is equal to (n - 1) / 2
res is equal to s[start]
let integer l is equal to start, r is equal to start
while ((l is greater than or equal to 0) or (r is less than n))
decrease l value by 1
increase r value by 1
if (r is less than n) , res is equal to res + rth element of s
if (l is greater than or equal to 0) , res is equal to res + lth element of s
print res and endline"""
        
        # Text area for pseudocode input
        pseudocode = st.text_area(
            "Enter your pseudocode:",
            value=example_pseudocode,
            height=300,
            help="Write clear, step-by-step pseudocode instructions",
            placeholder="Type your pseudocode here..."
        )
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            generate_btn = st.button("🚀 Generate Code", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
    
    with col2:
        st.subheader("💻 Output: Generated Code")
        
        if generate_btn and pseudocode.strip():
            with st.spinner("⚡ Generating code..."):
                start_time = time.time()
                
                # Generate code
                generated_code = generate_code(
                    model, tokenizer, device, 
                    pseudocode, max_length, temperature, num_beams
                )
                
                generation_time = time.time() - start_time
                
                # Display results
                if generated_code.startswith("Error"):
                    st.markdown('<div class="error-box">❌ <b>Generation Failed</b></div>', unsafe_allow_html=True)
                    st.error(generated_code)
                else:
                    st.markdown(f'<div class="success-box">✅ <b>Code Generated Successfully!</b><br>⏱️ Time: {generation_time:.2f} seconds</div>', unsafe_allow_html=True)
                    
                    # Code display
                    st.code(generated_code, language='cpp', line_numbers=True)
                    
                    # Download button
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        st.download_button(
                            label="📥 Download as .cpp",
                            data=generated_code,
                            file_name="generated_code.cpp",
                            mime="text/x-c++src",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        st.download_button(
                            label="📥 Download as .txt",
                            data=generated_code,
                            file_name="generated_code.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
        else:
            st.markdown('<div class="info-box">👈 <b>Ready to generate!</b><br>Enter pseudocode on the left and click "Generate Code"</div>', unsafe_allow_html=True)
    
    # Examples section
    st.markdown("---")
    st.markdown("## 🎯 Example Pseudocode Patterns")
    st.markdown("*Try these examples to see how the model works!*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📥 **Input/Output**")
        st.code("""read integer n
print n""", language="text")
        
        st.markdown("### 🔄 **Loops**")
        st.code("""read integer n
for i from 1 to n
    print i""", language="text")
    
    with col2:
        st.markdown("### ❓ **Conditionals**")
        st.code("""read integers a, b
if a greater than b
    print a
else
    print b""", language="text")
        
        st.markdown("### 📊 **Arrays**")
        st.code("""read integer n
create array a of size n
for i from 0 to n-1
    read a[i]""", language="text")
    
    with col3:
        st.markdown("### 🔤 **Strings**")
        st.code("""create string s
read s
for each character c in s
    print c""", language="text")
        
        st.markdown("### 🧮 **Math**")
        st.code("""read integers a, b
let sum = a + b
let prod = a * b
print sum, prod""", language="text")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><b>🚀 SPoC: Pseudocode to Code Converter</b></p>
        <p>Built with Streamlit • Powered by GPT-2 + LoRA • Fine-tuned on SPoC Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()