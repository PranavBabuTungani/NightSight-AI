import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.block(x)

class NightSightUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool=nn.MaxPool2d(2)
        self.enc1=DoubleConv(3,32)
        self.enc2=DoubleConv(32,64)
        self.enc3=DoubleConv(64,128)
        self.bottleneck=DoubleConv(128,256)
        self.up3=nn.ConvTranspose2d(256,128,2,2)
        self.dec3=DoubleConv(256,128)
        self.up2=nn.ConvTranspose2d(128,64,2,2)
        self.dec2=DoubleConv(128,64)
        self.up1=nn.ConvTranspose2d(64,32,2,2)
        self.dec1=DoubleConv(64,32)
        self.output=nn.Conv2d(32,3,1)
        self.activation=nn.Sigmoid()
    def forward(self,x):
        e1=self.enc1(x)
        e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2))
        b=self.bottleneck(self.pool(e3))
        d3=self.dec3(torch.cat([self.up3(b),e3],dim=1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],dim=1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],dim=1))
        return self.activation(self.output(d1))

st.set_page_config(page_title="NightSight AI",page_icon="🌙",layout="wide")

st.title("🌙 NightSight AI")
st.subheader("Deep Learning Based Low-Light Image Enhancement")
st.write("Enhance dark photographs using our trained U-Net model.")

@st.cache_resource
def load_model():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=NightSightUNet().to(device)
    checkpoint=torch.load("nightsight_best.pth",map_location=device,weights_only=True)
    state=checkpoint.get("model_state_dict",checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model,device

try:
    model,device=load_model()
except Exception as e:
    st.error("Unable to load the model. Check that nightsight_best.pth is in the app folder.")
    st.exception(e)
    st.stop()

uploaded=st.file_uploader("Upload a low-light photograph",type=["jpg","jpeg","png"])

gamma=st.slider("Gamma correction value",min_value=0.2,max_value=1.0,value=0.6,step=0.05)

if uploaded is not None:
    image=Image.open(uploaded).convert("RGB")
    st.image(image,caption="Original photograph",use_container_width=True)

    if st.button("✨ Enhance Image",type="primary"):
        with st.spinner("Enhancing your photograph..."):
            original_size=image.size
            model_input=image.resize((256,256),Image.Resampling.LANCZOS)
            tensor=transforms.ToTensor()(model_input).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction=model(tensor).squeeze(0).cpu().clamp(0,1)

            enhanced_array=(prediction.permute(1,2,0).numpy()*255).round().astype(np.uint8)
            enhanced=Image.fromarray(enhanced_array).resize(original_size,Image.Resampling.LANCZOS)

            original_array=np.asarray(image,dtype=np.float32)/255.0
            gamma_array=np.power(original_array,gamma)
            gamma_image=Image.fromarray((np.clip(gamma_array,0,1)*255).round().astype(np.uint8))

        st.success("Enhancement complete!")
        col1,col2,col3=st.columns(3)
        with col1:
            st.image(image,caption="Original",use_container_width=True)
        with col2:
            st.image(gamma_image,caption=f"Gamma correction ({gamma})",use_container_width=True)
        with col3:
            st.image(enhanced,caption="NightSight AI",use_container_width=True)

        st.download_button(
            "⬇️ Download enhanced image",
            data=__import__("io").BytesIO(
                (lambda b:(enhanced.save(b,format="PNG"),b.getvalue())[1])(__import__("io").BytesIO())
            ).getvalue(),
            file_name="nightsight_enhanced.png",
            mime="image/png"
        )

st.divider()
st.caption("NightSight AI | PyTorch U-Net | Low-Light Image Enhancement")
