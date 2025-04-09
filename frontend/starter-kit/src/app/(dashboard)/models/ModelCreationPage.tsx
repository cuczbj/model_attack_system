"use client";

import React, { useState, useCallback, useEffect } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  Stepper,
  Step,
  StepLabel,
  Paper,
  CircularProgress,
  Alert,
  AlertTitle,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Card,
  CardContent,
  Divider,
  Snackbar,
  IconButton,
  styled
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import RefreshIcon from "@mui/icons-material/Refresh";

// API基础URL
const API_URL = "http://localhost:5000";

// 数据类型定义
interface ModelFileStatus {
  hasStructure: boolean;
  hasParameters: boolean;
  structureFile?: string;
  parameterFiles: string[];
}

interface ModelConfig {
  model_name: string;
  param_file: string;
  class_num: number;
  input_shape: [number, number];
  model_type: string;
}

// 自定义样式组件
const StyledDropzone = styled(Box)(({ theme }) => ({
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(4),
  textAlign: "center",
  color: theme.palette.text.secondary,
  transition: "border 0.3s ease-in-out",
  height: "180px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  "&:hover": {
    borderColor: theme.palette.primary.dark,
  },
}));

// 步骤组件
const StepOne = ({ modelName, setModelName, onNext }) => {
  const [error, setError] = useState("");

  const handleNext = () => {
    if (!modelName.trim()) {
      setError("请输入模型名称");
      return;
    }
    
    // 验证模型名称格式
    if (!/^[A-Za-z0-9_]+$/.test(modelName)) {
      setError("模型名称只能包含字母、数字和下划线");
      return;
    }
    
    setError("");
    onNext();
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h6" gutterBottom>
        第一步：输入模型名称
      </Typography>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        请输入您想要创建的模型名称。系统将检查是否已存在相应的模型文件。
      </Typography>
      
      <TextField
        fullWidth
        label="模型名称"
        value={modelName}
        onChange={(e) => setModelName(e.target.value)}
        error={!!error}
        helperText={error}
        placeholder="例如: MLP, VGG16, FaceNet64"
        margin="normal"
        sx={{ mb: 3 }}
      />
      
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleNext}
          endIcon={<ArrowForwardIcon />}
        >
          下一步
        </Button>
      </Box>
    </Box>
  );
};

const StepTwo = ({ modelName, fileStatus, onRefresh, onNext, onBack, onUploadSuccess }) => {
  const [uploadingType, setUploadingType] = useState<"structure" | "parameter" | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [notification, setNotification] = useState({ open: false, message: "", severity: "success" as "success" | "error" });

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
      setUploadError("");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadError("请先选择文件");
      return;
    }

    // 验证文件名称是否符合规范
    if (uploadingType === "structure") {
      if (file.name !== `${modelName}.py`) {
        setUploadError(`模型结构文件名必须为 ${modelName}.py`);
        return;
      }
    } else if (uploadingType === "parameter") {
      if (!file.name.startsWith(`${modelName}_`) || !file.name.match(/\.(pth|tar|pkl)$/)) {
        setUploadError(`参数文件名必须以 ${modelName}_ 开头，并以 .pth、.tar 或 .pkl 结尾`);
        return;
      }
    }

    setUploading(true);
    setUploadProgress(0);
    setUploadError("");

    // 创建FormData对象
    const formData = new FormData();
    formData.append("file", file);

    // 创建XMLHttpRequest对象以支持进度监控
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/checkpoint`, true);

    // 监控上传进度
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const progress = Math.round((event.loaded / event.total) * 100);
        setUploadProgress(progress);
      }
    };

    xhr.onload = () => {
      setUploading(false);
      if (xhr.status === 200) {
        // 上传成功
        setFile(null);
        setUploadingType(null);
        setNotification({
          open: true,
          message: "文件上传成功！",
          severity: "success"
        });
        
        // 通知父组件上传成功
        onUploadSuccess();
      } else {
        try {
          const response = JSON.parse(xhr.responseText);
          setUploadError(response.error || "上传失败");
        } catch (e) {
          setUploadError("上传失败: " + xhr.statusText);
        }
      }
    };

    xhr.onerror = () => {
      setUploading(false);
      setUploadError("网络错误，上传失败");
    };

    xhr.send(formData);
  };

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  // 计算是否可以进入下一步
  const canProceed = fileStatus.hasStructure && fileStatus.hasParameters;

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h6" gutterBottom>
        第二步：检查并上传模型文件
      </Typography>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        系统已检查是否存在相应的模型文件。如果缺少任何文件，请按照规范上传。
      </Typography>

      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ mb: 2 }}>
            <strong>{modelName}</strong> 模型文件状态
          </Typography>
          
          {/* 模型结构文件状态 */}
          <Alert 
            severity={fileStatus.hasStructure ? "success" : "warning"}
            icon={fileStatus.hasStructure ? <CheckCircleIcon /> : <ErrorIcon />}
            sx={{ mb: 2 }}
          >
            <AlertTitle>模型结构文件</AlertTitle>
            {fileStatus.hasStructure 
              ? `已找到模型结构文件: ${fileStatus.structureFile}`
              : `未找到模型结构文件，请上传 ${modelName}.py`}
          </Alert>
          
          {/* 模型参数文件状态 */}
          <Alert 
            severity={fileStatus.hasParameters ? "success" : "warning"}
            icon={fileStatus.hasParameters ? <CheckCircleIcon /> : <ErrorIcon />}
            sx={{ mb: 2 }}
          >
            <AlertTitle>模型参数文件</AlertTitle>
            {fileStatus.hasParameters 
              ? `已找到${fileStatus.parameterFiles.length}个匹配的参数文件: ${fileStatus.parameterFiles.join(", ")}`
              : `未找到参数文件，请上传以 ${modelName}_ 开头的 .pth、.tar 或 .pkl 文件`}
          </Alert>
          
          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button 
              startIcon={<RefreshIcon />}
              onClick={onRefresh}
              size="small"
            >
              刷新状态
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* 文件上传区域 */}
      {(!fileStatus.hasStructure || !fileStatus.hasParameters) && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            上传缺失的文件
          </Typography>

          {!uploadingType ? (
            <Grid container spacing={2}>
              {!fileStatus.hasStructure && (
                <Grid item xs={12} md={6}>
                  <Button 
                    variant="outlined" 
                    fullWidth 
                    startIcon={<CloudUploadIcon />}
                    onClick={() => setUploadingType("structure")}
                    sx={{ p: 2 }}
                  >
                    上传模型结构文件 ({modelName}.py)
                  </Button>
                </Grid>
              )}
              
              {!fileStatus.hasParameters && (
                <Grid item xs={12} md={6}>
                  <Button 
                    variant="outlined" 
                    fullWidth 
                    startIcon={<CloudUploadIcon />}
                    onClick={() => setUploadingType("parameter")}
                    sx={{ p: 2 }}
                  >
                    上传模型参数文件 ({modelName}_*.pth/tar/pkl)
                  </Button>
                </Grid>
              )}
            </Grid>
          ) : (
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  {uploadingType === "structure" 
                    ? `上传模型结构文件 (${modelName}.py)` 
                    : `上传模型参数文件 (${modelName}_*.pth/tar/pkl)`}
                </Typography>
                
                <Alert severity="info" sx={{ mb: 2 }}>
                  <AlertTitle>文件命名规范</AlertTitle>
                  {uploadingType === "structure" 
                    ? `模型结构文件必须命名为 ${modelName}.py`
                    : `参数文件必须以 ${modelName}_ 开头，扩展名为 .pth、.tar 或 .pkl`}
                </Alert>
                
                <input
                  accept={uploadingType === "structure" ? ".py" : ".pth,.tar,.pkl"}
                  style={{ display: 'none' }}
                  id="file-upload-input"
                  type="file"
                  onChange={handleFileChange}
                />
                
                <label htmlFor="file-upload-input">
                  <StyledDropzone 
                    sx={{ 
                      height: "120px", 
                      cursor: 'pointer',
                      border: file ? '2px solid #4caf50' : undefined
                    }}
                  >
                    <CloudUploadIcon style={{ fontSize: 40, marginBottom: 8 }} color="primary" />
                    {file ? (
                      <Typography variant="body1" color="primary">
                        已选择: {file.name}
                      </Typography>
                    ) : (
                      <Typography variant="body1">
                        点击选择文件
                      </Typography>
                    )}
                  </StyledDropzone>
                </label>
                
                {uploadError && (
                  <Alert severity="error" sx={{ mt: 2 }}>
                    {uploadError}
                  </Alert>
                )}
                
                {uploading && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" gutterBottom>
                      上传进度: {uploadProgress}%
                    </Typography>
                    <LinearProgress variant="determinate" value={uploadProgress} />
                  </Box>
                )}
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                  <Button 
                    variant="outlined"
                    onClick={() => {
                      setUploadingType(null);
                      setFile(null);
                      setUploadError("");
                    }}
                    disabled={uploading}
                  >
                    取消
                  </Button>
                  
                  <Button 
                    variant="contained"
                    color="primary"
                    onClick={handleUpload}
                    disabled={!file || uploading}
                    startIcon={uploading ? <CircularProgress size={20} /> : null}
                  >
                    {uploading ? "上传中..." : "上传文件"}
                  </Button>
                </Box>
              </CardContent>
            </Card>
          )}
        </Box>
      )}

      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
        <Button
          variant="outlined"
          onClick={onBack}
          startIcon={<ArrowBackIcon />}
        >
          上一步
        </Button>
        
        <Button
          variant="contained"
          color="primary"
          onClick={onNext}
          endIcon={<ArrowForwardIcon />}
          disabled={!canProceed}
        >
          下一步
        </Button>
      </Box>
      
      <Snackbar
        open={notification.open}
        autoHideDuration={5000}
        onClose={handleCloseNotification}
      >
        <Alert onClose={handleCloseNotification} severity={notification.severity}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

const StepThree = ({ modelName, fileStatus, onBack, onComplete }) => {
  const [formData, setFormData] = useState<Partial<ModelConfig>>({
    model_name: modelName,
    param_file: fileStatus.parameterFiles.length > 0 ? fileStatus.parameterFiles[0] : "",
    class_num: 40, // 默认分类数
    input_shape: [112, 92] as [number, number], // 默认输入形状
    model_type: modelName,
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  
  // 处理输入形状变化
  const handleInputShapeChange = (index: number, value: number) => {
    const newShape = [...formData.input_shape] as [number, number];
    newShape[index] = value;
    setFormData({ ...formData, input_shape: newShape });
  };
  
  // 处理表单字段变化
  const handleChange = (field: string, value: any) => {
    setFormData({ ...formData, [field]: value });
  };
  
  // 创建模型配置
  const handleCreateModel = async () => {
    try {
      setLoading(true);
      setError("");
      
      // 调用API创建模型配置
      const response = await fetch(`${API_URL}/api/models/configurations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || "创建模型失败");
      }
      
      // 创建成功，进入完成步骤
      onComplete(data.config);
      
    } catch (error) {
      console.error("创建模型出错:", error);
      setError(error.message || "创建模型时出错");
    } finally {
      setLoading(false);
    }
  };
  
  // 根据模型类型设置默认配置
  useEffect(() => {
    let defaultClassNum = 40;
    let defaultInputShape: [number, number] = [112, 92];
    
    // 根据模型名称设置不同的默认值
    if (modelName.includes("VGG") || modelName.includes("FaceNet") || modelName.includes("IR152")) {
      defaultClassNum = 1000;
      defaultInputShape = [64, 64];
    }
    
    setFormData({
      ...formData,
      class_num: defaultClassNum,
      input_shape: defaultInputShape,
    });
  }, [modelName]);
  
  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h6" gutterBottom>
        第三步：配置模型参数
      </Typography>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        请为您的模型配置以下参数。这些参数将确定模型的输入输出形状。
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <TextField
            fullWidth
            label="模型名称"
            value={formData.model_name}
            disabled
            helperText="模型名称不可更改"
          />
        </Grid>
        
        <Grid item xs={12}>
          <FormControl fullWidth>
            <InputLabel id="param-file-label">参数文件</InputLabel>
            <Select
              labelId="param-file-label"
              value={formData.param_file}
              label="参数文件"
              onChange={(e) => handleChange("param_file", e.target.value)}
            >
              {fileStatus.parameterFiles.map((file) => (
                <MenuItem key={file} value={file}>
                  {file}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="分类数量"
            type="number"
            value={formData.class_num}
            onChange={(e) => handleChange("class_num", parseInt(e.target.value) || 0)}
            InputProps={{ inputProps: { min: 1 } }}
            helperText={
              modelName.includes("MLP") 
                ? "AT&T Faces数据集默认为40类" 
                : (modelName.includes("VGG") || modelName.includes("FaceNet") || modelName.includes("IR"))
                  ? "CelebA数据集默认为1000类"
                  : "请输入模型的分类数量"
            }
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>
            输入形状:
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="高度"
                type="number"
                value={formData.input_shape[0]}
                onChange={(e) => handleInputShapeChange(0, parseInt(e.target.value) || 0)}
                InputProps={{ inputProps: { min: 1 } }}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="宽度"
                type="number"
                value={formData.input_shape[1]}
                onChange={(e) => handleInputShapeChange(1, parseInt(e.target.value) || 0)}
                InputProps={{ inputProps: { min: 1 } }}
              />
            </Grid>
          </Grid>
          <Typography variant="caption" color="textSecondary">
            {modelName.includes("MLP") 
              ? "MLP模型默认输入形状为 112×92" 
              : (modelName.includes("VGG") || modelName.includes("FaceNet") || modelName.includes("IR"))
                ? "图像模型默认输入形状为 64×64"
                : "请设置合适的输入形状"}
          </Typography>
        </Grid>
      </Grid>
      
      {error && (
        <Alert severity="error" sx={{ mt: 3 }}>
          {error}
        </Alert>
      )}
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
        <Button
          variant="outlined"
          onClick={onBack}
          startIcon={<ArrowBackIcon />}
        >
          上一步
        </Button>
        
        <Button
          variant="contained"
          color="primary"
          onClick={handleCreateModel}
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : null}
        >
          {loading ? "创建中..." : "创建模型"}
        </Button>
      </Box>
    </Box>
  );
};

const StepFour = ({ createdModel, onReset }) => {
  return (
    <Box sx={{ mt: 4, textAlign: 'center' }}>
      <CheckCircleIcon color="success" sx={{ fontSize: 80 }} />
      
      <Typography variant="h5" sx={{ mt: 2, mb: 3 }}>
        模型创建成功！
      </Typography>
      
      <Card variant="outlined" sx={{ mb: 4, mx: 'auto', maxWidth: 600 }}>
        <CardContent>
          <Typography variant="h6" align="center" gutterBottom>
            {createdModel.model_name}
          </Typography>
          
          <Divider sx={{ my: 2 }} />
          
          <Grid container spacing={2}>
            <Grid item xs={6} textAlign="right">
              <Typography variant="body2" color="textSecondary">模型ID:</Typography>
            </Grid>
            <Grid item xs={6} textAlign="left">
              <Typography variant="body2">{createdModel.model_id}</Typography>
            </Grid>
            
            <Grid item xs={6} textAlign="right">
              <Typography variant="body2" color="textSecondary">参数文件:</Typography>
            </Grid>
            <Grid item xs={6} textAlign="left">
              <Typography variant="body2">{createdModel.param_file}</Typography>
            </Grid>
            
            <Grid item xs={6} textAlign="right">
              <Typography variant="body2" color="textSecondary">分类数量:</Typography>
            </Grid>
            <Grid item xs={6} textAlign="left">
              <Typography variant="body2">{createdModel.class_num}</Typography>
            </Grid>
            
            <Grid item xs={6} textAlign="right">
              <Typography variant="body2" color="textSecondary">输入形状:</Typography>
            </Grid>
            <Grid item xs={6} textAlign="left">
              <Typography variant="body2">{createdModel.input_shape.join(' × ')}</Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
      
      <Typography variant="body1" paragraph>
        您可以在模型管理页面查看和管理此模型，或者在攻击页面中使用此模型。
      </Typography>
      
      <Box sx={{ mt: 3 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={onReset}
        >
          创建新模型
        </Button>
      </Box>
    </Box>
  );
};

// 模型创建页面主组件
const ModelCreationPage: React.FC = () => {
  // 状态定义
  const [activeStep, setActiveStep] = useState(0);
  const [modelName, setModelName] = useState("");
  const [fileStatus, setFileStatus] = useState<ModelFileStatus>({
    hasStructure: false,
    hasParameters: false,
    parameterFiles: []
  });
  const [loading, setLoading] = useState(false);
  const [createdModel, setCreatedModel] = useState<ModelConfig | null>(null);
  const [error, setError] = useState("");

  // 清空模型数据
  const handleReset = () => {
    setActiveStep(0);
    setModelName("");
    setFileStatus({
      hasStructure: false,
      hasParameters: false,
      parameterFiles: []
    });
    setCreatedModel(null);
    setError("");
  };

  // 检查模型文件状态
  const checkModelFiles = useCallback(async () => {
    if (!modelName) return;
    
    try {
      setLoading(true);
      setError("");
      
      // 使用search API检查模型文件状态
      const response = await fetch(`${API_URL}/api/models/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model_name: modelName }),
      });
      
      if (!response.ok) {
        throw new Error("检查模型文件状态失败");
      }
      
      const data = await response.json();
      
      setFileStatus({
        hasStructure: data.structure_found,
        hasParameters: data.parameters_found,
        structureFile: data.structure_found ? `${modelName}.py` : undefined,
        parameterFiles: data.matching_parameters || []
      });
      
    } catch (error) {
      console.error("检查模型文件出错:", error);
      setError("检查模型文件状态时出错: " + error.message);
    } finally {
      setLoading(false);
    }
  }, [modelName]);

  // 步骤切换函数
  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  // 当进入第二步时，检查模型文件状态
  useEffect(() => {
    if (activeStep === 1) {
      checkModelFiles();
    }
  }, [activeStep, checkModelFiles]);

  // 步骤完成回调
  const handleComplete = (model: ModelConfig) => {
    setCreatedModel(model);
    setActiveStep(3); // 直接跳到最后一步
  };

  // 渲染步骤内容
  const renderStepContent = () => {
    switch (activeStep) {
      case 0:
        return (
          <StepOne 
            modelName={modelName} 
            setModelName={setModelName} 
            onNext={handleNext} 
          />
        );
      case 1:
        return (
          <StepTwo 
            modelName={modelName}
            fileStatus={fileStatus}
            onRefresh={checkModelFiles}
            onNext={handleNext}
            onBack={handleBack}
            onUploadSuccess={checkModelFiles}
          />
        );
      case 2:
        return (
          <StepThree 
            modelName={modelName}
            fileStatus={fileStatus}
            onBack={handleBack}
            onComplete={handleComplete}
          />
        );
      case 3:
        return (
          <StepFour 
            createdModel={createdModel}
            onReset={handleReset}
          />
        );
      default:
        return null;
    }
  };

  const steps = ['输入模型名称', '检查/上传模型文件', '配置模型参数', '完成'];

  return (
    <Box sx={{ maxWidth: 900, margin: '0 auto', p: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        创建新模型
      </Typography>
      
      <Typography variant="body1" paragraph align="center" sx={{ mb: 4 }}>
        通过简单步骤创建并配置机器学习模型，系统会自动检测所需文件并引导您完成配置过程。
      </Typography>
      
      <Paper sx={{ p: 3, mb: 4 }}>
        <Stepper activeStep={activeStep} alternativeLabel>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Paper>
      
      {loading && activeStep !== 3 ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 5 }}>
          <CircularProgress />
          <Typography sx={{ mt: 2 }}>加载中...</Typography>
        </Box>
      ) : (
        <Paper sx={{ p: 3 }}>
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}
          
          {renderStepContent()}
        </Paper>
      )}
      
      <Box sx={{ mt: 4, bgcolor: '#f5f5f5', p: 2, borderRadius: 1 }}>
        <Typography variant="subtitle1" gutterBottom>
          文件命名规范
        </Typography>
        <Typography variant="body2">
          1. <strong>模型结构文件</strong>必须使用<code>ModelName.py</code>格式，例如:
        </Typography>
        <Typography variant="body2" component="div" sx={{ pl: 3 }}>
          • MLP.py<br />
          • VGG16.py<br />
          • FaceNet64.py<br />
          • IR152.py
        </Typography>
        <Typography variant="body2" sx={{ mt: 1 }}>
          2. <strong>模型参数文件</strong>必须以<code>ModelName_</code>开头，后面可以添加描述信息，扩展名可以是<code>.pth</code>、<code>.tar</code>或<code>.pkl</code>，例如:
        </Typography>
        <Typography variant="body2" component="div" sx={{ pl: 3 }}>
          • MLP_ATT40.pkl（MLP模型，训练于AT&T 40类数据集）<br />
          • VGG16_CelebA1000.tar（VGG16模型，训练于CelebA 1000类数据集）<br />
          • IR152_Face.pth（IR152模型，训练于人脸数据集）
        </Typography>
      </Box>
    </Box>
  );
};

export default ModelCreationPage;
