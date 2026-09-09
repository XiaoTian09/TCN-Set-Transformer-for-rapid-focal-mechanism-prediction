# Publish this directory

Create an empty repository in your GitHub account. From this release's root,
substitute your account/repository URL in the final command:

```bash
git init -b main
git add .
git status --short
git commit -m "Release focal-mechanism code and trained model"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Add the intended code/model license and citation metadata under your authorship.
The package contains only the requested model binary and selected code/docs; no
training dataset is needed for upload. `.gitignore` keeps datasets and outputs out
of subsequent additions. The included ~14 MB checkpoint can be committed as a
normal Git binary; no LFS pointer or external model download is required.

The repository root is the directory containing `README.md`, not `INSTANCE_test`
and not its entire parent `github_release` directory. The ZIP is an alternative
transfer artifact containing the same release files. This preparation did not
initialize a remote repository or publish anything.
