"use client";

import React, { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import AccordionDetails from "@mui/material/AccordionDetails";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import Divider from "@mui/material/Divider";
import Button from "@mui/material/Button";
import Link from "@mui/material/Link";
import Paper from "@mui/material/Paper";

const HelpSection = ({ title, children }) => {
  return (
    <Accordion defaultExpanded>
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        aria-controls={`panel-${title}-content`}
        id={`panel-${title}-header`}
      >
        <Typography variant="h6">{title}</Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Box>{children}</Box>
      </AccordionDetails>
    </Accordion>
  );
};

const HelpDocumentation = () => {
  return (
    <Box
      style={{
        padding: "20px",
        borderRadius: "8px",
        backgroundColor: "#f5f5f5",
        maxWidth: "800px",
        margin: "0 auto",
        boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
      }}
    >
      <Typography variant="h5" gutterBottom>
        帮助文档
      </Typography>
      <Typography variant="body1" paragraph>
      本系统专为机器学习安全研究人员、数据隐私分析师和模型开发者设计，提供了一套完整的工具来评估机器学习分类模型对隐私攻击的抵抗能力。系统核心功能是通过多种逆向工程技术，尝试从目标模型中重建原始训练数据，从而揭示潜在的隐私泄露风险。
      </Typography>

      <Divider style={{ margin: "20px 0" }} />

      <HelpSection title="系统概述">
        <Typography variant="body1" paragraph>
          本系统通过多种攻击方法，对机器学习分类模型进行逆向工程，尝试重建原始训练数据。主要功能包括：
        </Typography>
        <ul style={{ paddingLeft: "20px" }}>
          <li>
            <Typography variant="body1" paragraph>
              <strong>模型上传与管理</strong>：支持本地模型上传
            </Typography>
          </li>
          <li>
            <Typography variant="body1" paragraph>
              <strong>参数配置</strong>：调整攻击参数，包括分类数、迭代次数、学习率等
            </Typography>
          </li>
          <li>
            <Typography variant="body1" paragraph>
              <strong>攻击方法</strong>：提供多种攻击策略，包括模型反演、成员推理等
            </Typography>
          </li>
          <li>
            <Typography variant="body1" paragraph>
              <strong>结果可视化</strong>：直观展示攻击结果，包括重建图像和评估指标
            </Typography>
          </li>
        </ul>
      </HelpSection>

      <HelpSection title="使用流程">
        <Box component="ol" sx={{ paddingLeft: "20px" }}>
          <li>
            <Typography variant="body1" paragraph>
              <strong>输入数据与模型</strong>：上传目标模型或提供API接口
            </Typography>
          </li>
          <li>
            <Typography variant="body1" paragraph>
              <strong>参数设置</strong>：配置攻击参数，包括分类数、迭代次数、学习率等
            </Typography>
          </li>
          <li>
            <Typography variant="body1" paragraph>
              <strong>选择攻击方法</strong>：从多种攻击策略中选择合适的方法
            </Typography>
          </li>
          <li>
            <Typography variant="body1" paragraph>
              <strong>执行攻击</strong>：系统自动执行攻击过程，并实时反馈攻击进度
            </Typography>
          </li>
          <li>
            <Typography variant="body1" paragraph>
              <strong>查看结果</strong>：分析攻击效果，查看重建的图像和评估指标
            </Typography>
          </li>
        </Box>
      </HelpSection>

      <HelpSection title="攻击方法说明">
        <Typography variant="subtitle1" gutterBottom>
          本系统支持以下几种攻击方法：
        </Typography>

        <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            1.
          </Typography>
          <Typography variant="body2" paragraph>
            
          </Typography>
        </Paper>

        <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            2.
          </Typography>
          <Typography variant="body2" paragraph>
           
          </Typography>
        </Paper>

        <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            GAN辅助重建 (GAN-based Reconstruction)
          </Typography>
          <Typography variant="body2" paragraph>
            结合生成对抗网络技术，通过优化GAN的隐空间，生成与目标类别相符的高质量图像。相比基础模型反演攻击，生成的图像质量更高。
          </Typography>
        </Paper>


      </HelpSection>

      <HelpSection title="参数配置指南">
        <Typography variant="body1" paragraph>
          配置合适的参数对攻击效果有显著影响，以下是主要参数的说明：
        </Typography>

        <Box sx={{ ml: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            分类数
          </Typography>
          <Typography variant="body2" paragraph>
            目标模型的分类数量。对于多分类模型，需要准确设置分类数以保证攻击有效性。
          </Typography>

          <Typography variant="subtitle1" gutterBottom>
            迭代次数
          </Typography>
          <Typography variant="body2" paragraph>
            攻击算法的迭代次数。较高的迭代次数通常能提高重建质量，但会增加计算时间。建议值范围：100-1000。
          </Typography>

          <Typography variant="subtitle1" gutterBottom>
            学习率 (攻击强度)
          </Typography>
          <Typography variant="body2" paragraph>
            控制每次迭代的步长。较高的学习率可能导致更快的收敛，但也可能导致结果不稳定。建议初始值：0.1-0.5。
          </Typography>
        </Box>
      </HelpSection>



      <Divider style={{ margin: "20px 0" }} />

      <Box style={{ display: "flex", justifyContent: "center", gap: "20px" }}>
        <Button variant="contained" color="primary" href="/">
          返回首页
        </Button>
        <Button
          variant="outlined"
          color="secondary"
          href="mailto:support@example.com"
        >
          联系技术支持
        </Button>
      </Box>

      <Box sx={{ mt: 4, textAlign: "center" }}>
        <Typography variant="caption" color="textSecondary">
          如需更多帮助，请参考
          <Link href="#" color="primary" sx={{ ml: 1 }}>
            技术文档
          </Link>
          或
          <Link href="#" color="primary" sx={{ ml: 1, mr: 1 }}>
            常见问题解答
          </Link>
        </Typography>
      </Box>
    </Box>
  );
};

export default function Page() {
  return (
    <div>
      <h1>机器学习模型攻击系统</h1>
      <div style={{ marginTop: "30px", marginBottom: "30px" }}></div>
      <HelpDocumentation />
    </div>
  );
}
