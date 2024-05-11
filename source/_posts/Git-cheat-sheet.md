---
title: Git cheat sheet
date: 2023-12-31 15:32:59
categories:
- work
tags:
- git
---

[Pro Git Book](https://git-scm.com/book/zh/v2)

<!--more-->

## 起步

### Git特色

**直接记录快照，而非差异比较**
与其他版本控制系统基于差异的版本控制不同，Git直接记录文件快照，每次提交或保存都会基于当前的全部文件创建一个快照并保存这个快照的索引。为了高效，如果文件没有变化，Git不会再次保存，而是只保留一个链接指向之前存储的文件。

**近乎所有操作都在本地执行**

**保证完整性**
Git中所有数据在存储前都会计算校验和（SHA-1），并以校验和的哈希值作为引用和索引。

**只添加数据**
几乎所有操作都只是添加，几乎没有导致文件不可恢复的操作。

**三种状态**
- modified：已修改，文件在工作区被修改，但还未保存到数据库中
- staged：已暂存，修改的文件已经被做了标记，将被包含在下次提交中
- commited：已提交，已经安全的保存在本地数据库中

### 运行前配置 git config

配置文件优先级由低到高
- --system:针对该系统上所有用户的所有仓库的通用配置，一般在Git安装目录
- --global:当前用户的所有仓库，一般在用户目录
- --local:当前仓库，一般在当前仓库目录

> git config --global user.name "John Doe"  
> git config --global user.email johndoe@example.com  
> git config --list  # 可能会有重复，因为Git会依次读取系统、全局、当前仓库的config，后面的会覆盖前面的同名配置


## Cheat Sheet

### Remote
查看所有分支：`branch` -r远程，-a所有，-v附带详细信息

将本地的分支推送至远程仓库：`push <remote> <local_branch>:<remote_branch>`

删除某远程分支：`push <remote> --delete <branch_name>`

### Local
查看提交记录：`log` --pretty=oneline指定以一行的形式显示

比较差异：`diff <hash1> <hash2>` 

增删分支：`branch <branch_name>` -d删除 -t追踪远程分支

获取远程仓库数据：`fetch` 从远程仓库下载数据，但不会自动合并，后续需要merge或pull

撤销提交：`reset` --soft仅仅撤销提交，--mixed撤销提交并取消暂存区，--hard撤销提交并取消暂存区和工作区 HEAD~n表示前n个版本

### Index/Stage
提交到本地仓库：`commit` -a自动将所有已跟踪文件暂存起来并提交，相当于先执行add -u ，-m指定提交信息，--amend修改最后一次提交

比较当前index与某次提交的差异：`diff --cached <hash>`

### Workspace
检查状态：`status` -s简洁

克隆远程仓库：`clone` 创建一个全新副本

从远程仓库拉取最新文件：`pull` 相当于fetch+merge

添加文件至Index：`add` -u更新已跟踪文件（即不包括新增文件），-A更新所有改动（包括新增文件），.当前目录所有文件

签出某个分支或提交或文件：`checkout` -b创建并签出新分支。如果签出的是一个特定提交，会处于"detached HEAD"状态，可以随意修改并提交，但这个提交不属于任何分支（但仍然存在，只要记得提交的哈希值），可以通过创建分支来保存

将某一分支的所有更改合并到当前分支：`merge <branch_name>`，Git默认使用Fast forward模式，即如果被合并分支是当前分支的直接后继，Git会将当前分支指向被合并分支的最新提交并不产生新提交；这时用--no-ff禁用Fast forward模式，仍然会产生一次新提交。如果并非直接前后继关系，Git会创建一个新的合并提交，该提交有两个父提交，即两个分支的最新提交。

变基：`rebase` 一般不用。

将其他分支的某个提交应用到当前分支：`cherry-pick <hash>`，会产生一个新的提交，提交内容（记得git是保存修改后文件的快照）与原提交相同。

回滚某提交：`revert <hash>`，会产生一个新的提交，提交内容与原提交相反。

清理未被跟踪的文件：`clean` -n列出将要被清理的文件，-f清理文件，-d清理文件夹，-x清理忽略的文件，-X清理忽略的文件夹

临时保存工作区和暂存区，将这些改动进栈：`stash` -u保存未被跟踪的文件，-a保存所有改动，-p交互式保存，-k不保存暂存区的改动

恢复工作区和暂存区：`stash pop` 恢复并删除栈顶的stash，`stash apply` 恢复但不删除栈顶的stash

### Stash
显示栈：`stash list`

入栈：`stash push` 默认入栈顶

显示栈的细节：`stash show` 默认栈顶

删除当个栈：`stash drop` 默认栈顶

清空整个stash：`stash clear`