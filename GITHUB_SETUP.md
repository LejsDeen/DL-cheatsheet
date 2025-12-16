# Quick GitHub Setup Guide

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `cheat_sheet` (or whatever you prefer)
3. Description: "Typst cheat sheet for [Your Course/Exam]"
4. Choose **Public** (for collaboration) or **Private** (if you prefer)
5. **DO NOT** check "Initialize this repository with a README" (we already have one)
6. Click "Create repository"

## Step 2: Connect Local Repo to GitHub

After creating the repository on GitHub, you'll see a page with setup instructions. Use these commands:

```bash
# Replace YOUR_USERNAME and YOUR_REPO_NAME with your actual values
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

**Example:**
If your GitHub username is `johndoe` and your repo is `dl-exam-cheatsheet`:
```bash
git remote add origin https://github.com/johndoe/dl-exam-cheatsheet.git
git branch -M main
git push -u origin main
```

## Step 3: Verify Setup

After pushing, refresh your GitHub repository page. You should see:
- ✅ All your files (cheat_sheet.typ, README.md, Makefile, .gitignore)
- ✅ Commit history

## Enabling Collaboration

### Adding Collaborators

1. Go to your repository on GitHub
2. Click **Settings** → **Collaborators** → **Add people**
3. Enter the GitHub username or email of your collaborators
4. They'll receive an invitation to collaborate

### Alternative: Using GitHub Organizations/Teams

If you're working with a class or group, consider:
- Creating a GitHub Organization for your class
- Using GitHub Classroom (if your instructor uses it)

## Next Steps

- Share the repository URL with your collaborators
- Make sure everyone has Typst CLI installed
- Start collaborating! See README.md for collaboration workflow

