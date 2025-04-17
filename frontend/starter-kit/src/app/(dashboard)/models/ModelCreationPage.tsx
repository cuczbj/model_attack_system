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
  styled,
  LinearProgress,
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
  model_id?: string;
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

const StepThree = ({ modelName, fileStatus, onBack, onNext, onComplete }) => {
  // 表单数据状态 - 初始不设置param_file
  const [formData, setFormData] = useState({
    model_name: modelName,
    param_file: "",  // 初始为空
    class_num: 40, // 默认分类数
    input_shape: [112, 92], // 默认输入形状
    model_type: modelName,
  });
  
  // 上传数据集相关状态
  const [datasetFile, setDatasetFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState("");
  const [validationResult, setValidationResult] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  
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
  
  // 处理输入形状变化
  const handleInputShapeChange = (index, value) => {
    const newShape = [...formData.input_shape];
    newShape[index] = value;
    setFormData({ ...formData, input_shape: newShape as [number, number] });
  };
  
  // 处理表单字段变化
  const handleChange = (field, value) => {
    setFormData({ ...formData, [field]: value });
  };
  
  // 文件选择处理函数
  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setDatasetFile(event.target.files[0]);
      setError("");
    }
  };
  
  // 验证配置
  const validateConfig = async () => {
    if (!formData.model_name || !formData.param_file) {
      setValidationResult({
        valid: false,
        message: "请选择模型和参数文件"
      });
      return false;
    }
    
    try {
      const response = await fetch(`${API_URL}/api/models/validate-config`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });
      
      const data = await response.json();
      const result = {
        valid: response.ok,
        message: data.message || (response.ok ? "配置有效" : "配置无效")
      };
      
      setValidationResult(result);
      return result.valid;
    } catch (error) {
      setValidationResult({
        valid: false,
        message: "验证配置时出错: " + (error.message || "未知错误")
      });
      return false;
    }
  };
  
  // 上传数据集并创建模型
  const handleUploadAndCreate = async () => {
    // 首先验证配置是否有效
    const isValid = await validateConfig();
    if (!isValid) {
      return;
    }
    
    if (!datasetFile) {
      setError("请选择隐私数据集文件");
      return;
    }
    
    setUploading(true);
    setUploadProgress(0);
    setError("");
    
    try {
      // 创建FormData对象用于文件上传
      const uploadFormData = new FormData();
      uploadFormData.append("dataset_file", datasetFile);
      uploadFormData.append("dataset_name", `${modelName}_dataset`);
      uploadFormData.append("model_name", modelName);
      uploadFormData.append("description", `${modelName}模型的隐私数据集`);
      
      // 创建XMLHttpRequest以监控上传进度
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_URL}/api/datasets/upload-for-model`, true);
      
      // 监控上传进度
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(progress);
        }
      };
      
      // 设置上传完成回调
      xhr.onload = async () => {
        if (xhr.status === 200 || xhr.status === 201) {
          try {
            // 解析上传结果
            const uploadResult = JSON.parse(xhr.responseText);
            
            // 数据集上传成功后，创建模型配置
            setIsCreating(true);
            
            // 准备模型配置数据，包含数据集ID
            const modelConfigData = {
              model_name: formData.model_name,
              param_file: formData.param_file,
              class_num: formData.class_num,
              input_shape: formData.input_shape,
              model_type: formData.model_type || formData.model_name,
              dataset_id: uploadResult.id
            };
            
            // 调用API创建模型配置
            const configResponse = await fetch(`${API_URL}/api/models/configurations`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify(modelConfigData),
            });
            
            const configData = await configResponse.json();
            
            if (!configResponse.ok) {
              throw new Error(configData.error || "创建模型配置失败");
            }
            
            // 完成整个流程
            onComplete(configData.config);
          } catch (error) {
            setError("创建模型配置失败: " + (error.message || "未知错误"));
          } finally {
            setIsCreating(false);
          }
        } else {
          try {
            const response = JSON.parse(xhr.responseText);
            setError(response.error || "上传数据集失败");
          } catch (e) {
            setError("上传数据集失败");
          }
        }
        
        setUploading(false);
      };
      
      // 错误处理
      xhr.onerror = () => {
        setUploading(false);
        setError("网络错误，上传失败");
      };
      
      // 发送请求
      xhr.send(uploadFormData);
    } catch (error) {
      setUploading(false);
      setError("上传数据集时出错: " + (error.message || "未知错误"));
    }
  };
  
  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h6" gutterBottom>
        第三步：配置模型参数并上传隐私数据集
      </Typography>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        请配置模型参数并上传对应的隐私数据集。数据集应按类别组织，每个文件夹名称对应模型的一个分类标签。
      </Typography>
      
      <Grid container spacing={3}>
        {/* 模型配置部分 */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              模型配置
            </Typography>
            
            <TextField
              fullWidth
              label="模型名称"
              value={formData.model_name}
              disabled
              margin="normal"
              helperText="模型名称不可更改"
            />
            
            <FormControl fullWidth margin="normal">
              <InputLabel id="param-file-label">参数文件</InputLabel>
              <Select
                labelId="param-file-label"
                value={formData.param_file}
                label="参数文件"
                onChange={(e) => handleChange("param_file", e.target.value)}
              >
                {fileStatus?.parameterFiles?.map((file) => (
                  <MenuItem key={file} value={file}>
                    {file}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <TextField
              fullWidth
              label="分类数量"
              type="number"
              value={formData.class_num}
              onChange={(e) => handleChange("class_num", parseInt(e.target.value) || 0)}
              margin="normal"
              InputProps={{ inputProps: { min: 1 } }}
              helperText={
                formData.model_name.includes("MLP") 
                  ? "AT&T Faces数据集默认为40类" 
                  : (formData.model_name.includes("VGG") || formData.model_name.includes("FaceNet") || formData.model_name.includes("IR"))
                    ? "CelebA数据集默认为1000类"
                    : "请输入模型的分类数量"
              }
            />
            
            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
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
              {formData.model_name.includes("MLP") 
                ? "MLP模型默认输入形状为 112×92" 
                : (formData.model_name.includes("VGG") || formData.model_name.includes("FaceNet") || formData.model_name.includes("IR"))
                  ? "图像模型默认输入形状为 64×64"
                  : "请设置合适的输入形状"}
            </Typography>
            
            <Button
              variant="outlined"
              color="primary"
              onClick={validateConfig}
              disabled={uploading || isCreating}
              sx={{ mt: 2 }}
            >
              验证配置
            </Button>
            
            {validationResult && (
              <Alert 
                severity={validationResult.valid ? "success" : "error"}
                sx={{ mt: 1 }}
              >
                {validationResult.message}
              </Alert>
            )}
          </Paper>
        </Grid>
        
        {/* 数据集上传部分 */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              上传隐私数据集
            </Typography>
            
            <Alert severity="info" sx={{ mb: 2 }}>
              <AlertTitle>数据集格式要求</AlertTitle>
              <Typography variant="body2">
                • 支持的格式：ZIP压缩文件<br/>
                • 目录结构：根目录下应包含按类别命名的子文件夹（如"0"、"1"、"2"等）<br/>
                • 每个子文件夹中包含该类别的隐私图像文件<br/>
                • 示例："5"文件夹中包含所有第5类的隐私图像
              </Typography>
            </Alert>
            
            <Box
              sx={{
                border: '2px dashed',
                borderColor: 'primary.main',
                borderRadius: 1,
                p: 3,
                mb: 2,
                textAlign: 'center',
                cursor: 'pointer'
              }}
              onClick={() => document.getElementById('dataset-upload-input').click()}
            >
              <CloudUploadIcon color="primary" sx={{ fontSize: 48, mb: 1 }} />
              
              {datasetFile ? (
                <Typography variant="body1" color="primary">
                  已选择: {datasetFile.name}
                </Typography>
              ) : (
                <Typography variant="body1">
                  点击选择数据集文件
                </Typography>
              )}
              
              <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 1 }}>
                支持 .zip、.tar.gz、.tar 格式
              </Typography>
            </Box>
            
            <input
              accept=".zip,.tar.gz,.tar"
              style={{ display: 'none' }}
              id="dataset-upload-input"
              type="file"
              onChange={handleFileChange}
            />
            
            {uploading && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="body2">
                  上传进度: {uploadProgress}%
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={uploadProgress} 
                  sx={{ height: 8, borderRadius: 1, mt: 1 }}
                />
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
      
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
        <Button
          variant="outlined"
          onClick={onBack}
          startIcon={<ArrowBackIcon />}
          disabled={uploading || isCreating}
        >
          上一步
        </Button>
        
        <Button
          variant="contained"
          color="primary"
          onClick={handleUploadAndCreate}
          disabled={!datasetFile || uploading || isCreating}
          endIcon={
            uploading || isCreating ? 
            <CircularProgress size={20} color="inherit" /> : 
            null
          }
        >
          {uploading ? "上传中..." : isCreating ? "创建中..." : "上传并创建模型"}
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
            {createdModel?.model_name}
          </Typography>
          
          <Divider sx={{ my: 2 }} />
          
          <Grid container spacing={2}>
            <Grid item xs={6} textAlign="right">
              <Typography variant="body2" color="textSecondary">模型ID:</Typography>
            </Grid>
            <Grid item xs={6} textAlign="left">
              <Typography variant="body2">{createdModel?.model_id}</Typography>
            </Grid>
            
            <Grid item xs={6} textAlign="right">
              <Typography variant="body2" color="textSecondary">参数文件:</Typography>
            </Grid>
            <Grid item xs={6} textAlign="left">
              <Typography variant="body2">{createdModel?.param_file}</Typography>
            </Grid>
            
            <Grid item xs={6} textAlign="right">
              <Typography variant="body2" color="textSecondary">分类数量:</Typography>
            </Grid>
            <Grid item xs={6} textAlign="left">
              <Typography variant="body2">{createdModel?.class_num}</Typography>
            </Grid>
            
            <Grid item xs={6} textAlign="right">
              <Typography variant="body2" color="textSecondary">输入形状:</Typography>
            </Grid>
            <Grid item xs={6} textAlign="left">
              <Typography variant="body2">{createdModel?.input_shape.join(' × ')}</Typography>
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
const ModelCreationPage = ({ onComplete }) => {
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
  const [datasetId, setDatasetId] = useState(null);

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
    setDatasetId(null);
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
    
    // 通知父组件创建完成
    if (onComplete) {
      onComplete(model);
    }
  };

  // 渲染步骤内容
// 更新renderStepContent函数
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
      onNext={(datasetId) => {
        setDatasetId(datasetId);
        handleNext();
      }}
      onBack={handleBack}
      onComplete={handleComplete}  // 添加这一行
    />
  );
    case 3:
      return (
        <StepFour 
          modelName={modelName}
          fileStatus={fileStatus}
          datasetId={datasetId}
          onBack={handleBack}
          onComplete={handleComplete}
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