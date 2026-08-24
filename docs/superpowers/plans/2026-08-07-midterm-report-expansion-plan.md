# 中期考核报告扩充实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将中期考核报告扩展为不少于15页的正式文档，同时保持当前研究口径、模板版式和信息真实性。

**Architecture:** 以现有已填写DOCX为版式基础，以开题报告提供背景、意义和文献现状，以中期PPT提供当前问题定义、技术方案、阶段成果和计划。通过Python-docx生成扩展版DOCX，再由Word转换并覆盖指定DOC，最后使用Word导出PDF并逐页渲染验收。

**Tech Stack:** Python 3、python-docx、Microsoft Word COM、pypdfium2、PowerShell。

## Global Constraints

- 最终文档不少于15页，目标15—16页。
- 不虚构实验数值、论文发表、软件著作权授权、导师意见和专家结论。
- 已完成、进行中和计划开展必须明确区分。
- 保持封面、填表说明、导师意见和专家考核栏的原始模板样式。
- 最终覆盖 `D:\Intelligent_Scheduling\中期答辩\硕士研究生论文中期考核报告（2024） - 副本.doc`，同时保留备份及DOCX版本。

---

### Task 1: 构建扩充版正文内容

**Files:**
- Modify: `D:\Intelligent_Scheduling\.codex_tmp\build_midterm_report.py`
- Read: `D:\Intelligent_Scheduling\开题答辩\硕士研究生选题报告（2024）(1).doc`
- Read: `D:\Intelligent_Scheduling\中期答辩\修订版.pptx`

**Interfaces:**
- Consumes: 开题报告中的研究背景、意义、文献现状、目标、方法和参考文献；中期PPT中的最新研究定义和阶段进展。
- Produces: 结构完整、事实边界清晰的第二部分正文与四类表格数据。

- [ ] **Step 1: 建立内容覆盖检查**

检查扩写脚本必须包含以下标题：研究背景、研究意义、国内外研究现状、研究目标与内容调整、研究内容与关键问题、技术路线、已完成工作与阶段成果、当前不足、实验设计、成果情况、后续计划、阶段总结。

- [ ] **Step 2: 扩写背景、意义和国内外现状**

从开题报告提炼背景与意义，增加两张文献现状表；所有结论改写为当前“任务需求 + 资源可行性 + 运行状态”口径。

- [ ] **Step 3: 扩写中期核心内容**

详细说明问题一、问题二、输入变量S/T/R/wT、候选策略评价、示范预训练、PPO反馈优化、已完成原型与未完成模块。

- [ ] **Step 4: 生成扩展DOCX**

运行：

```powershell
& 'C:\Users\33758\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'D:\Intelligent_Scheduling\.codex_tmp\build_midterm_report.py'
```

预期：生成 `硕士研究生论文中期考核报告（2024）-按开题报告完善.docx`，命令退出码为0。

### Task 2: 检查内容真实性与结构完整性

**Files:**
- Test: `D:\Intelligent_Scheduling\中期答辩\硕士研究生论文中期考核报告（2024）-按开题报告完善.docx`

**Interfaces:**
- Consumes: Task 1生成的DOCX。
- Produces: 内容检查结果，包括标题覆盖、表格数量、禁止性表述和空白签字栏。

- [ ] **Step 1: 提取正文并检查标题覆盖**

使用python-docx提取正文和表格文本，要求12个章节标题全部存在，SimPy、Gym、PPO和12种候选策略均被正确描述。

- [ ] **Step 2: 检查事实边界**

确认文档未出现已获得实验优势、已发表论文、已获软件著作权、导师同意或专家通过等无依据结论。

- [ ] **Step 3: 检查表格结构**

确认培养计划表、两张文献现状表、阶段成果表和后续计划表均存在；表头完整，行数符合内容设计。

### Task 3: 分页与视觉迭代

**Files:**
- Modify: `D:\Intelligent_Scheduling\.codex_tmp\build_midterm_report.py`
- QA: `C:\Users\33758\AppData\Local\Temp\codex-docs\midterm-expanded-*`

**Interfaces:**
- Consumes: 内容检查通过的DOCX。
- Produces: 15—16页且无视觉缺陷的DOCX/PDF。

- [ ] **Step 1: 使用Word导出PDF并统计页数**

通过Word COM打开DOCX、重新分页并导出PDF。页数低于15页时增加必要论证内容或调整合理段落间距；高于16页时压缩重复内容，不通过缩小到难以阅读的字号解决。

- [ ] **Step 2: 将PDF逐页渲染为PNG**

运行：

```powershell
& 'C:\Users\33758\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'D:\Intelligent_Scheduling\.codex_tmp\render_pdf_pages.py' '<PDF路径>' '<PNG目录>'
```

预期：PNG数量与Word统计页数一致。

- [ ] **Step 3: 逐页视觉检查**

检查所有页面无文字裁切、表格断裂、标题孤立、边框错位、乱码和大面积异常空白；发现问题后修改脚本并重新生成、导出、渲染。

### Task 4: 生成并验证最终DOC

**Files:**
- Modify: `D:\Intelligent_Scheduling\中期答辩\硕士研究生论文中期考核报告（2024） - 副本.doc`
- Preserve: `D:\Intelligent_Scheduling\中期答辩\硕士研究生论文中期考核报告（2024）-原始备份.doc`
- Create: `D:\Intelligent_Scheduling\中期答辩\硕士研究生论文中期考核报告（2024）-扩充版.docx`

**Interfaces:**
- Consumes: Task 3通过视觉验收的DOCX。
- Produces: 指定路径下的最终DOC及可编辑DOCX副本。

- [ ] **Step 1: 保存最终DOCX并转换DOC**

使用Word COM将最终DOCX另存为临时DOC，随后覆盖用户指定的副本文件；若文件被占用，先输出独立DOC并等待占用释放后重试。

- [ ] **Step 2: 重新打开最终DOC验证**

验证页数不少于15页、所有章节和表格存在、导师和专家栏保持空白、文件可正常打开。

- [ ] **Step 3: 最终PDF渲染对比**

重新导出最终DOC的PDF并逐页渲染，确认与已验收DOCX在页数和版式上保持一致。

- [ ] **Step 4: 报告交付位置**

仅交付最终DOC和可编辑DOCX，说明原始备份位置以及仍需用户补充的开题报告学分、导师意见和专家信息。
