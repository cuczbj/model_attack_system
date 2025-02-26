"use client";

import React, { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Slider from "@mui/material/Slider";
import Divider from "@mui/material/Divider";

const ParameterConfig = () => {
  const [parameters, setParameters] = useState({
    numClasses: 10, // 分类数
    iterations: 100, // 迭代次数
    attackStrength: 0.5, // 攻击强度
  });

  // 更新参数状态
  const handleChange = (name: string, value: number) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // 重置参数
  const resetParameters = () => {
    setParameters({
      numClasses: 10,
      iterations: 100,
      attackStrength: 0.5,
    });
  };

  return (
    <Box
      style={{
        padding: "20px",
        borderRadius: "8px",
        backgroundColor: "#f5f5f5",
        maxWidth: "600px",
        margin: "0 auto",
        boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
      }}
    >
      <Typography variant="h5" gutterBottom>
        参数配置
      </Typography>
      <Typography variant="body1" gutterBottom>
        设置模型攻击参数，调整分类数、迭代次数以及攻击强度。
      </Typography>

      <Divider style={{ margin: "20px 0" }} />

      {/* 分类数设置 */}
      <Box style={{ marginBottom: "20px" }}>
        <Typography gutterBottom>分类数：</Typography>
        <TextField
          type="number"
          value={parameters.numClasses}
          onChange={(e) =>
            handleChange("numClasses", Math.max(1, parseInt(e.target.value) || 0))
          }
          fullWidth
          label="分类数 (≥1)"
          variant="outlined"
        />
      </Box>

      {/* 迭代次数设置 */}
      <Box style={{ marginBottom: "20px" }}>
        <Typography gutterBottom>迭代次数：</Typography>
        <TextField
          type="number"
          value={parameters.iterations}
          onChange={(e) =>
            handleChange("iterations", Math.max(1, parseInt(e.target.value) || 0))
          }
          fullWidth
          label="迭代次数 (≥1)"
          variant="outlined"
        />
      </Box>

      {/* 攻击强度设置 */}
      <Box style={{ marginBottom: "20px" }}>
        <Typography gutterBottom>攻击强度：</Typography>
        <Slider
          value={parameters.attackStrength}
          onChange={(e, value) =>
            handleChange("attackStrength", value as number)
          }
          step={0.1}
          min={0}
          max={1}
          valueLabelDisplay="auto"
        />
        <Typography variant="caption" color="textSecondary">
          当前攻击强度：{parameters.attackStrength.toFixed(2)}
        </Typography>
      </Box>

      <Divider style={{ margin: "20px 0" }} />

      {/* 操作按钮 */}
      <Box style={{ display: "flex", justifyContent: "space-between" }}>
        <Button
          variant="contained"
          color="primary"
          onClick={() => alert(`参数已保存：\n${JSON.stringify(parameters)}`)}
        >
          保存参数
        </Button>
        <Button variant="outlined" color="secondary" onClick={resetParameters}>
          重置
        </Button>
      </Box>
    </Box>
  );
};

export default function Page() {
  return (
    <div>
      <h1>机器学习模型攻击系统</h1>
      <div style={{ marginTop: "30px", marginBottom: "30px" }}></div>
      <ParameterConfig />
    </div>
  );
}
