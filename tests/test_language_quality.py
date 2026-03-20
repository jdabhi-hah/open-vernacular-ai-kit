from __future__ import annotations

import pytest

from open_vernacular_ai_kit.codemix_render import render_codemix
from open_vernacular_ai_kit.token_lid import TokenLang, detect_token_lang


@pytest.mark.parametrize(
    ("token", "language"),
    [
        ("mane", "gu"),
        ("amne", "gu"),
        ("tamaro", "gu"),
        ("aapdu", "gu"),
        ("gayu", "gu"),
        ("chhiye", "gu"),
        ("aavo", "gu"),
        ("pachi", "gu"),
        ("tamare", "gu"),
        ("aa", "gu"),
        ("fari", "gu"),
        ("vage", "gu"),
        ("jagyae", "gu"),
        ("ochhu", "gu"),
        ("paise", "gu"),
        ("moklo", "gu"),
        ("batave", "gu"),
        ("avsho", "gu"),
        ("mujhe", "hi"),
        ("tumhara", "hi"),
        ("jayenge", "hi"),
        ("bahut", "hi"),
        ("parivar", "hi"),
        ("dhanyavad", "hi"),
        ("dijiye", "hi"),
        ("chahiye", "hi"),
        ("madad", "hi"),
        ("lekin", "hi"),
        ("galat", "hi"),
        ("batayiye", "hi"),
    ],
)
def test_detect_token_lang_expanded_language_hints(token: str, language: str) -> None:
    assert detect_token_lang(token, language=language) == TokenLang.TARGET_ROMAN


def test_short_context_tokens_do_not_become_global_target_words() -> None:
    assert detect_token_lang("me", language="hi") == TokenLang.EN
    assert detect_token_lang("ka", language="hi") == TokenLang.EN
    assert detect_token_lang("ma", language="gu") == TokenLang.EN
    assert detect_token_lang("ne", language="gu") == TokenLang.EN


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mane tari vaat samajh nathi padti", "મને તારી વાત સમજ નથી પડતી"),
        ("tamne aaje office ma aavu chhe", "તમને આજે office માં આવું છે"),
        ("aapdu kaam saras rite thai gayu", "આપણું કામ સરસ રીતે થઈ ગયું"),
        ("shu tame mane madad kari shako?", "શું તમે મને મદદ કરી શકો?"),
        ("amne ahi badma aavu joie", "અમને અહીં બાદમાં આવું જોઈએ"),
        ("tamaro parivar kya chhe?", "તમારો પરિવાર ક્યાં છે?"),
        ("ame kale amdavad ma chhiye", "અમે કાલે અમદાવાદ માં છીએ"),
        ("tame savare ahi aavo", "તમે સવારે અહીં આવો"),
        ("tame sanje tya jao", "તમે સાંજે ત્યાં જાઓ"),
        ("tamare ahi aavu joie", "તમારે અહીં આવું જોઈએ"),
        ("aa ghar tyaa chhe", "આ ઘર ત્યાં છે"),
        ("aa refund ni vaat chhe", "આ refund ની વાત છે"),
        ("aa refund ni jagyae replacement joie", "આ refund ની જગ્યાએ replacement જોઈએ"),
        ("payment pending chhe ke complete", "payment pending છે કે complete"),
        ("aaje customer care sathe vaat thai hati", "આજે customer care સાથે વાત થઈ હતી"),
        ("mari payment pending batave chhe", "મારી payment pending બતાવે છે"),
        ("delivery partner ne exact location moklo", "delivery partner ne exact location મોકલો"),
        ("aa coupon code kem kaam nathi karto", "આ coupon code કેમ કામ નથી કરતો"),
        ("mobile number change karvu chhe", "mobile number change કરવું છે"),
        ("customer care vala mane callback kare", "customer care vala મને callback કરે"),
        ("cod order mate exact cash ready rakhjo", "cod order માટે exact cash ready રાખજો"),
        ("tamaro otp expire thai gayo ke haju valid chhe", "તમારો otp expire થઈ gayo કે હજુ valid છે"),
        ("bank account verify thayu chhata payout atkyu chhe", "bank account verify થયું છતાં payout અટક્યું છે"),
        ("tamaro promo code minimum amount vagar kaam nathi karto", "તમારો promo code minimum amount વગર કામ નથી કરતો"),
        ("order hold mathi release kyare thase", "order hold માંથી release ક્યારે થશે"),
        ("cash memo pdf turant share karo", "cash memo pdf તુરંત share કરો"),
        ("cod order mate exact cash ready rakho please", "cod order માટે exact cash ready રાખો please"),
        ("nearest branch ma document physical submit karva padse", "nearest branch માં document physical submit karva પડશે"),
        ("tamaro plan auto renew off kem nathi thato", "તમારો plan auto renew off કેમ નથી થતો"),
        ("callback sanje j karjo ok?", "callback સાંજે j કરજો ok?"),
        ("office ma chu pachi call karjo", "office માં છું પછી call કરજો"),
        ("invoice pdf mail krjo pls", "invoice pdf mail કરજો pls"),
        ("pickup savare aavse ke bapor pachi?", "pickup સવારે આવશે કે બપોર પછી?"),
        ("parcel kya sudhi pohnchyoo?", "parcel ક્યાં સુધી પહોંચ્યો?"),
        ("verify mate photo dubara upload krdo", "verify માટે photo ફરીથી upload krdo"),
        ("payment fail thayu pn paisa cut thai gaya", "payment fail થયું pn પૈસા cut થઈ ગયા"),
        ("refund initiate thai gyo pn sms nathi aavyo", "refund initiate થઈ gyo pn sms નથી આવ્યો"),
        ("nearest store javun pade?", "nearest store જવું pade?"),
        ("wallet ma paisa ochha batave chhe", "wallet માં પૈસા ઓછા બતાવે છે"),
        ("mara refund vishe kal pan msg karyo hato", "મારા refund વિશે kal pan msg karyo હતો"),
        ("aa parcel building niche muki gaya ke shu?", "આ parcel building નીચે muki ગયા કે શું?"),
        ("tamari side thi koi callback aavyo nathi haju", "તમારી side થી koi callback આવ્યો નથી હજુ"),
        ("tamne invoice ni pic moklu to chale?", "તમને invoice ની pic moklu તો ચાલે?"),
        ("warehouse sudhi parcel pohchi gayu ke nahi?", "warehouse સુધી parcel pohchi ગયું કે nahi?"),
        ("screenshot moklyo chhe ema error batave chhe", "screenshot મોકલ્યો છે ema error બતાવે છે"),
        ("aa chat export ma badha msg chhe tame joi lo", "આ chat export માં બધા msg છે તમે joi lo"),
        ("mne otp 2 var try kari pachi pan nathi malyo", "મને otp 2 var try કરી પછી pan નથી મળ્યો"),
        ("group ma je number mukyo hato e par koi uthadto nathi", "group માં જે number mukyo હતો e par koi uthadto નથી"),
        ("tamaro app crash thay chhe jyare payment karu chu", "તમારો app crash thay છે જ્યારે payment કરું છું"),
        ("exchange ni jagyae simple refund joiye have", "exchange ની જગ્યાએ simple refund જોઈએ હવે"),
        ("wallet ma paisa gaya but order place nahi thayu", "wallet માં પૈસા ગયા but order place nahi થયું"),
        ("mne refund kyare aavse e j samajhatu nathi karan ke paisa cut thai gaya pan order confirm nathi thayu", "મને refund ક્યારે આવશે e j સમજાતું નથી karan કે પૈસા cut થઈ ગયા pan order confirm નથી થયું"),
        ("mara wallet mathi paisa gaya pachi app freeze thai gayu ane payment success batavyu pan order dekhatu nathi", "મારા wallet માંથી પૈસા ગયા પછી app freeze થઈ ગયું ane payment success બતાવ્યું pan order દેખાતું નથી"),
        ("cash memo joiye chhe karan ke reimbursement mate office ma submit karvu chhe jaldi moklo", "cash memo જોઈએ છે karan કે reimbursement માટે office માં submit કરવું છે જલ્દી મોકલો"),
        ("mara order ma ek item ochho aavyo chhe ane invoice ma pan be item j chhe tame check karo", "મારા order માં ek item ઓછો આવ્યો છે ane invoice માં pan બે item j છે તમે check કરો"),
        ("group ma je support number moklyo hato e par koi phone uthadvatu nathi to bijo number aapo", "group માં જે support number મોકલ્યો હતો e par koi phone uthadvatu નથી તો બીજો number આપો"),
        ("pickup boy aavyo hato pan kehva lagyo ke address adhuro chhe jyare hu badhu voice note ma kahi didhu hatu", "pickup boy આવ્યો હતો pan kehva lagyo કે address અધૂરો છે જ્યારે હું બધું voice note માં kahi દીધું hatu"),
        ("kyc screen shot ma surname blur dekhae chhe pn original clear htu", "kyc screen shot માં surname blur dekhae છે pn original clear હતું"),
        ("aa screenshot ma refund initiated lakhyu chhe pn paisa account ma nathi aavya", "આ screenshot માં refund initiated lakhyu છે pn પૈસા account માં નથી આવ્યા"),
        ("product label ni photo mathi color black ni jagyae blk j read thay chhe", "product label ની photo માંથી color black ની જગ્યાએ blk j read thay છે"),
    ],
)
def test_render_codemix_gujarati_quality_cases(raw: str, expected: str) -> None:
    assert render_codemix(raw, language="gu", translit_mode="sentence") == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mujhe tumse baat karni hai", "मुझे तुमसे बात करनी है"),
        ("aap kaise ho aur ghar kab aaoge", "आप कैसे हो और घर कब आओगे"),
        ("kal hum market jayenge", "कल हम market जाएंगे"),
        ("tumhara order aaj deliver hoga", "तुम्हारा order आज deliver होगा"),
        ("yeh bahut accha hai", "यह बहुत अच्छा है"),
        ("kahan ho tum", "कहाँ हो तुम"),
        ("meri maa ka naam kya hai", "मेरी माँ का नाम क्या है"),
        ("mera parivar kahan rehta hai", "मेरा परिवार कहाँ रहता है"),
        ("vah ghar me hai", "वह घर में है"),
        ("dhanyavad, main theek hun", "धन्यवाद, मैं ठीक हूँ"),
        ("mujhe paise dijiye", "मुझे पैसे दीजिए"),
        ("mujhe aap ki madad chahiye", "मुझे आप की मदद चाहिए"),
        ("aap hamare ghar aaiye", "आप हमारे घर आइए"),
        ("mujhe madad chahiye lekin samay nahin hai", "मुझे मदद चाहिए लेकिन समय नहीं है"),
        ("mujhe order status batayiye", "मुझे order status बताइए"),
        ("delivery late kyon ho rahi hai", "delivery late क्यों हो रही है"),
        ("invoice pdf mujhe whatsapp par bhejo", "invoice pdf मुझे whatsapp पर भेजो"),
        ("return pickup kal subah chahiye", "return pickup कल सुबह चाहिए"),
        ("coupon apply karyo but discount nahin mila", "coupon apply karyo but discount नहीं मिला"),
        ("delivery boy ghar ke niche wait kari raha hai", "delivery boy घर के नीचे wait kari रहा है"),
        ("cash memo pdf turant bhej dijiye", "cash memo pdf तुरंत भेज दीजिए"),
        ("nearest branch me document physically submit karna padega", "nearest branch में document physically submit करना पड़ेगा"),
        ("address proof clear nahin tha isliye reject hua", "address proof clear नहीं था इसलिए reject hua"),
        ("mujhe jaldi refund chahiye", "मुझे जल्दी refund चाहिए"),
        ("discount apply nhi hua aj bhi", "discount apply nhi hua आज bhi"),
        ("otp aj tk nahi aaya yrr", "otp आज तक नहीं aaya yrr"),
        ("office me hu baad me call krna", "office में hu बाद में call krna"),
        ("coupon code bilkul work nhi kr rha", "coupon code बिल्कुल work nhi kr rha"),
        ("verify ke liye photo dubara upload kr do", "verify के लिए photo दुबारा upload kr दो"),
        ("pickup subah hoga ya dopahar baad?", "pickup सुबह होगा ya दोपहर बाद?"),
        ("mere account se same amount do baar debit hua", "मेरे account से same amount दो बार debit hua"),
        ("mere order me ek item kam aaya hai aur invoice me bhi do hi item dikh rahe hain", "मेरे order में एक item kam aaya है और invoice में bhi दो hi item dikh रहे हैं"),
        ("group me jo support number bheja tha us par koi phone nahi utha raha to dusra number dijiye", "group में जो support number bheja था us पर koi phone नहीं utha रहा तो dusra number दीजिए"),
        ("exchange nahi simple refund chahiye kyunki main same item dubara nahi mangwana chahta", "exchange नहीं simple refund चाहिए क्योंकि मैं same item दुबारा नहीं mangwana chahta"),
    ],
)
def test_render_codemix_hindi_quality_cases(raw: str, expected: str) -> None:
    assert render_codemix(raw, language="hi", translit_mode="sentence") == expected
