from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://www.dropbox.com/s/gv3cvsmyd7uyk27/herbs-new-s3.pkl?raw=1'
export_file_name = 'herbs-new-s3.pkl'

classes = [
    'Chinese cabbage',
    'spinach',
    'choy sum',
    'ér cài ',
    'leaf mustard',
    'Chinese broccoli',
    'Tricolor daisy',
    'huáng xīn cài ',
    'fennel',
    'jī máo cài ',
    'garlic chives',
    'water spinach',
    'kuài cài ',
    'endive',
    'asparagus',
    'celery',
    'suàn huáng ',
    'garlic shoots',
    'crown daisy',
    'pea shoots',
    'lettuce',
    'cilantro',
    'parsley',
    'bok choy',
    'watercress',
    'lettuce',
    'oilseed rape',
    'kale',
    'bamboo shoot',
]

class_to_english = {
    '01': 'Chinese cabbage',
    '02': 'spinach',
    '03': 'choy sum',
    '04': 'ér cài\xa0',
    '05': 'leaf mustard',
    '06': 'Chinese broccoli',
    '07': 'Tricolor daisy',
    '08': 'huáng xīn cài\xa0',
    '09': 'fennel',
    '10': 'jī máo cài\xa0',
    '11': 'garlic chives',
    '12': 'water spinach',
    '13': 'kuài cài\xa0',
    '14': 'endive',
    '15': 'asparagus',
    '16': 'celery',
    '17': 'suàn huáng\xa0',
    '18': 'garlic shoots',
    '19': 'crown daisy',
    '20': 'pea shoots',
    '21': 'lettuce',
    '22': 'cilantro',
    '23': 'parsley',
    '24': 'bok choy',
    '25': 'watercress',
    '26': 'lettuce',
    '27': 'oilseed rape',
    '28': 'kale',
    '29': 'bamboo shoot',
}

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': class_to_english[str(prediction)]})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
