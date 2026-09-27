from dataset_loader import ChatDatasetLoader
##i have to insitanitate 
file="/home/saidharahas/gitprojects/gitrag/data/discord_chat2.json"
## how should i prroceed next
obj=ChatDatasetLoader(file)
#default 3
obj.load_messages()
chunks=obj.format_as_chunks()
for i in chunks :
     print(i)
