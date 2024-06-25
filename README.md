# Mochi Flashback
> A half-baked clone of Microsoft Recall, built in Python, for fun.

Mochi Flashback was created as part of a live
coding session on Twitch. The goal was to build a simple clone of Microsoft
and see how far i could get in a few hours across a week.  
_I have no intention to maintain this project._  

[Microsoft Recall](https://support.microsoft.com/en-us/windows/retrace-your-steps-with-recall-aa03f8a0-a78b-4b3e-b0a1-2eb8ac48701c)
is a new feature in the coming Windows that allows users to search across
any content they've seen on their PC.  
Microsoft takes regular screenshots and has them available for searching
with a natural language interface.  

## Streams
Follow me on [YouTube](https://www.youtube.com/@DiogoNeves) or [Twitch](https://www.twitch.tv/diogosnows).  
New content coming soon...  

[Day 1 to 5 of Mochi Flashback streams playlist](https://www.youtube.com/playlist?list=PLqFOswg8ElTKXaAtYWmWYpR0JY0uWgE1T)  
Follow the development of the code available in this repo.  
[![Mochi Flashback Playlist: Day 1 to 5](https://github.com/DiogoNeves/mochi-flashback/assets/178898/45373020-5f00-4ef6-a40c-fe8435995ec2)](https://www.youtube.com/playlist?list=PLqFOswg8ElTKXaAtYWmWYpR0JY0uWgE1T)
  
## 🙏 Thanks to
https://solara.dev/ for developing this nice framework!  

## Installation
> There are no promises made this would run and I have only tested on macOS (Macbook Air M1) with python 3.11.8
> I did not freeze the requirements either.

Create the virtual environment:  
```bash
# Recommended
$ python -m venv .venv
```

Install the dependencies:  
```bash
$ pip install solara pillow openai
```

## Running
### 1. Add screenshots
Screenshots should go into the `data` folder as seen in:  
![Screenshot Folder](https://github.com/DiogoNeves/mochi-flashback/assets/178898/9b28eab9-4bd8-4012-bf4d-e2686267a049)  

> ⚠️ Unfortunately I did not get to taking the screenshots automatically.

### 2. Process data
Process the data:  
```bash
$ python process_screenshots.py
```
This will create datastores with the processed data in either `local_stores` or `openai_stores` folders.  

### 3. Run recall
Here's were we use Solara:  
```bash
$ solara run recall.py
```

### 4. Interact with the UI
Here are sample results:  
![Results](https://github.com/DiogoNeves/mochi-flashback/assets/178898/423d16bf-c9e6-473a-aea4-5f3b0ecdb278)

## License
[LICENSE](https://github.com/DiogoNeves/mochi-flashback/LICENSE)
