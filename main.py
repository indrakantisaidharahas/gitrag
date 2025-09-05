import os
import git
# Path where you want the folder
path = input("enter the directory name u fucker ")

# Create directory (only if it doesn't exist)
os.makedirs(path, exist_ok=True)

print(f"Directory '{path}' created successfully")


repo_url = input("enter the repo url u bitch ,that u want to make a summarisation of :")
destination_path = "~gitprojects/gitrag/'{path}'"

try:
    git.Repo.clone_from(repo_url, destination_path)
    print(f"Repository cloned successfully to: {destination_path}")
except git.exc.GitCommandError as e:
    print(f"Error cloning repository: {e}")
''' i will make a recusrive that read a content of of the directories '''
def readdir(dirname) :
        content=oslistdir(dirname)
        for i in content :
              tpath=dirname+'/i'
              if(os.isdir(tpath)):
                   readdir(tpath)
              else:
                   tdf = pd.read_csv(tpath, sep='\s+', header=None, names=['col1', 'col2'])
                   df=pd.concat([df, tdf], ignore_index=True)
                   